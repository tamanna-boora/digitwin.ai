"""Pure line simulation: walks every unit through all 24 stations, emitting
sensor readings, manual checks, and ground-truth defects with their eventual
gate detection. No I/O — callers own writing the output anywhere.

Timeline model: production runs continuously across shifts within a day
(no overnight gap modeled), so `sim_time_hours` accumulates across the whole
horizon and shift assignment is derived by walking shift durations modulo
one production day. This keeps fault ramps (tool wear) and the humidity
series simple single continuous series instead of per-day resets.
"""

from dataclasses import dataclass

import numpy as np

from twinline.schemas import (
    DefectRecord,
    DefectType,
    InspectionGateConfig,
    ManualCheck,
    ModelConfig,
    PlantLineConfig,
    Reading,
    SensorNoiseConfig,
    ShiftConfig,
    ShiftId,
    SimulationOutput,
    StationConfig,
    UnitRecord,
    VariantConfig,
)
from twinline.sim.faults import FaultSchedule, build_fault_schedule, evaluate_station_faults

_AREA_FALLBACK_DEFECT_TYPE = {
    "BODY": DefectType.WELD_DEFECT,
    "PAINT": DefectType.PAINT_DEFECT,
    "FINAL_ASSEMBLY": DefectType.ASSEMBLY_DEFECT,
}


@dataclass(frozen=True)
class _DayShiftLayout:
    day_length_hours: float
    shift_order: list[tuple[ShiftId, float, float]]  # (id, start_offset_hours, end_offset_hours)


def _build_day_layout(shifts: list[ShiftConfig]) -> _DayShiftLayout:
    order: list[tuple[ShiftId, float, float]] = []
    cursor = 0.0
    for shift in shifts:
        duration = shift.end_hour - shift.start_hour
        order.append((shift.id, cursor, cursor + duration))
        cursor += duration
    return _DayShiftLayout(day_length_hours=cursor, shift_order=order)


def _shift_for_time(layout: _DayShiftLayout, sim_time_hours: float) -> ShiftId:
    offset = sim_time_hours % layout.day_length_hours
    for shift_id, start, end in layout.shift_order:
        if start <= offset < end:
            return shift_id
    return layout.shift_order[-1][0]


def total_production_hours(plant: PlantLineConfig, model: ModelConfig) -> float:
    layout = _build_day_layout(plant.shifts)
    return layout.day_length_hours * model.simulation_days


def total_units(plant: PlantLineConfig, model: ModelConfig) -> int:
    hours = total_production_hours(plant, model)
    return int(hours * 3600.0 / plant.takt_seconds)


def _pick_variant(variants: list[VariantConfig], rng: np.random.Generator) -> VariantConfig:
    ids = [v.id for v in variants]
    weights = [v.mix_ratio for v in variants]
    choice = rng.choice(len(ids), p=weights)
    return variants[choice]


def _sensor_reading_value(
    sensor_name: str,
    station: StationConfig,
    variant: VariantConfig,
    defect_here: bool,
    model: ModelConfig,
    rng: np.random.Generator,
) -> float:
    noise_cfg: SensorNoiseConfig = model.sensor_noise
    std_frac = noise_cfg.rich_std_frac if station.instrumentation.value == "rich" else noise_cfg.partial_std_frac

    if sensor_name == "cycle_time_s":
        nominal = station.base_cycle_time_s * variant.cycle_time_multiplier
        return float(rng.normal(nominal, nominal * std_frac))

    spec = model.sensor_specs[sensor_name]
    mean = spec.nominal * (1.0 + spec.defect_shift_frac) if defect_here else spec.nominal
    std = abs(spec.nominal) * std_frac
    return float(rng.normal(mean, std))


def _resolve_defect_type(station: StationConfig, contributed_types: list[DefectType]) -> DefectType:
    if contributed_types:
        return contributed_types[0]
    return _AREA_FALLBACK_DEFECT_TYPE[station.area.value]


def simulate_line(plant: PlantLineConfig, model: ModelConfig, rng: np.random.Generator) -> SimulationOutput:
    layout = _build_day_layout(plant.shifts)
    n_units = total_units(plant, model)
    horizon_hours = total_production_hours(plant, model)

    schedule: FaultSchedule = build_fault_schedule(
        model.fault_sources, model.ambient_humidity, horizon_hours, n_units, rng
    )

    units: list[UnitRecord] = []
    readings: list[Reading] = []
    manual_checks: list[ManualCheck] = []
    defects: list[DefectRecord] = []

    stations_by_sequence = sorted(plant.stations, key=lambda s: s.sequence)

    for seq in range(n_units):
        start_time_s = seq * plant.takt_seconds
        sim_time_hours = start_time_s / 3600.0
        shift_id = _shift_for_time(layout, sim_time_hours)
        variant = _pick_variant(plant.variants, rng)
        unit_id = f"UNIT-{seq:06d}"

        units.append(
            UnitRecord(
                unit_id=unit_id,
                variant_id=variant.id,
                shift_id=shift_id,
                sequence_number=seq,
                start_time_s=start_time_s,
            )
        )

        pending_defect: DefectRecord | None = None

        for station in stations_by_sequence:
            visit_time_s = start_time_s + station.sequence * plant.takt_seconds
            visit_time_hours = visit_time_s / 3600.0

            defect_created_here = False
            if pending_defect is None and station.can_cause_defect:
                contribution = evaluate_station_faults(
                    station.id, seq, visit_time_hours, shift_id, model.fault_sources, schedule
                )
                trigger_prob = model.base_defect_rate + contribution.added_rate
                if rng.random() < trigger_prob:
                    defect_created_here = True
                    defect_type = _resolve_defect_type(station, contribution.defect_types)
                    pending_defect = DefectRecord(
                        unit_id=unit_id,
                        origin_station_id=station.id,
                        defect_type=defect_type,
                        causes=contribution.causes,
                        created_time_s=visit_time_s,
                        detected=False,
                    )

            if station.instrumentation.value == "manual":
                operator_id = f"OP-{shift_id.value}-{(seq % 6) + 1:02d}"
                check_pass = not defect_created_here
                manual_checks.append(
                    ManualCheck(
                        unit_id=unit_id,
                        station_id=station.id,
                        timestamp_s=visit_time_s,
                        operator_id=operator_id,
                        check_pass=check_pass,
                    )
                )
            else:
                for sensor_name in station.sensors:
                    value = _sensor_reading_value(
                        sensor_name, station, variant, defect_created_here, model, rng
                    )
                    readings.append(
                        Reading(
                            unit_id=unit_id,
                            station_id=station.id,
                            timestamp_s=visit_time_s,
                            sensor_name=sensor_name,
                            value=value,
                        )
                    )

            if station.is_inspection_gate and pending_defect is not None:
                gate = plant.gate_for_defect_type(pending_defect.defect_type)
                if gate.station_id == station.id:
                    pending_defect = _resolve_detection(pending_defect, gate, plant.takt_seconds, rng)
                    defects.append(pending_defect)
                    pending_defect = None

        if pending_defect is not None:
            raise RuntimeError(
                f"defect on {unit_id} at {pending_defect.origin_station_id} never reached its inspection "
                "gate — a fault source's origin station must sequence before its gate"
            )

    return SimulationOutput(units=units, readings=readings, manual_checks=manual_checks, defects=defects)


def _resolve_detection(
    defect: DefectRecord,
    gate: InspectionGateConfig,
    takt_seconds: float,
    rng: np.random.Generator,
) -> DefectRecord:
    detected = rng.random() < gate.detection_probability
    if not detected:
        return defect.model_copy(update={"detected": False})

    lag_units = float(np.clip(rng.normal(gate.lag_mean_units, gate.lag_std_units), gate.lag_min_units, gate.lag_max_units))
    detection_time_s = defect.created_time_s + lag_units * takt_seconds
    return defect.model_copy(
        update={
            "detected": True,
            "detection_station_id": gate.station_id,
            "detection_time_s": detection_time_s,
            "gap_units": lag_units,
        }
    )
