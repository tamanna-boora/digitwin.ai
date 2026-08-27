"""Pure fault-source evaluation: given a schedule, how much defect probability
a station picks up at a given point in production. No I/O, no hidden state —
all randomness flows through the numpy Generator the caller provides.
"""

import math
from dataclasses import dataclass

import numpy as np

from twinline.schemas import AmbientHumidityConfig, DefectType, FaultKind, FaultSourceConfig, ShiftId


@dataclass(frozen=True)
class FaultSchedule:
    """Precomputed randomness (bad-batch draws, hourly humidity) for one sim run."""

    bad_batch_flags: dict[str, list[bool]]
    humidity_by_hour: list[float]


@dataclass(frozen=True)
class FaultContribution:
    added_rate: float
    causes: list[str]
    defect_types: list[DefectType]


def build_fault_schedule(
    fault_sources: list[FaultSourceConfig],
    ambient_cfg: AmbientHumidityConfig,
    total_hours: float,
    total_units: int,
    rng: np.random.Generator,
) -> FaultSchedule:
    bad_batch_flags: dict[str, list[bool]] = {}
    for source in fault_sources:
        if source.kind != FaultKind.SUPPLIER_BATCH:
            continue
        assert source.batch_size_units is not None and source.bad_batch_probability is not None
        n_batches = total_units // source.batch_size_units + 1
        draws = rng.random(n_batches)
        bad_batch_flags[source.id] = [bool(d < source.bad_batch_probability) for d in draws]

    n_hours = int(total_hours) + 2
    humidity_by_hour = _build_humidity_series(ambient_cfg, n_hours, rng)

    return FaultSchedule(bad_batch_flags=bad_batch_flags, humidity_by_hour=humidity_by_hour)


def _build_humidity_series(cfg: AmbientHumidityConfig, n_hours: int, rng: np.random.Generator) -> list[float]:
    hours = np.arange(n_hours)
    seasonal = cfg.amplitude_pct * np.sin(2.0 * math.pi * hours / cfg.period_hours)
    noise = rng.normal(0.0, cfg.noise_std_pct, size=n_hours)
    values = np.clip(cfg.mean_pct + seasonal + noise, 0.0, 100.0)
    return [float(v) for v in values]


def evaluate_station_faults(
    station_id: str,
    unit_seq: int,
    sim_time_hours: float,
    shift_id: ShiftId,
    fault_sources: list[FaultSourceConfig],
    schedule: FaultSchedule,
) -> FaultContribution:
    added_rate = 0.0
    causes: list[str] = []
    defect_types: list[DefectType] = []

    for source in fault_sources:
        if station_id not in source.station_ids:
            continue
        rate = _source_added_rate(source, unit_seq, sim_time_hours, shift_id, schedule)
        if rate <= 0.0:
            continue
        added_rate += rate
        causes.append(source.id)
        if source.defect_type not in defect_types:
            defect_types.append(source.defect_type)

    return FaultContribution(added_rate=added_rate, causes=causes, defect_types=defect_types)


def _source_added_rate(
    source: FaultSourceConfig,
    unit_seq: int,
    sim_time_hours: float,
    shift_id: ShiftId,
    schedule: FaultSchedule,
) -> float:
    if source.kind == FaultKind.TOOL_WEAR:
        return _tool_wear_rate(source, sim_time_hours)
    if source.kind == FaultKind.SUPPLIER_BATCH:
        return _supplier_batch_rate(source, unit_seq, schedule)
    if source.kind == FaultKind.OPERATOR_VARIATION:
        return _operator_variation_rate(source, shift_id)
    if source.kind == FaultKind.AMBIENT:
        return _ambient_rate(source, sim_time_hours, schedule)
    raise ValueError(f"unhandled fault kind {source.kind}")


def _tool_wear_rate(source: FaultSourceConfig, sim_time_hours: float) -> float:
    assert source.onset_hour is not None and source.ramp_per_hour is not None
    assert source.max_added_rate is not None
    if sim_time_hours < source.onset_hour:
        return 0.0
    elapsed_hours = sim_time_hours - source.onset_hour
    return min(elapsed_hours * source.ramp_per_hour, source.max_added_rate)


def _supplier_batch_rate(source: FaultSourceConfig, unit_seq: int, schedule: FaultSchedule) -> float:
    assert source.batch_size_units is not None and source.bad_batch_added_rate is not None
    batch_index = unit_seq // source.batch_size_units
    flags = schedule.bad_batch_flags[source.id]
    is_bad = flags[batch_index] if batch_index < len(flags) else False
    return source.bad_batch_added_rate if is_bad else 0.0


def _operator_variation_rate(source: FaultSourceConfig, shift_id: ShiftId) -> float:
    assert source.shift_multipliers is not None and source.added_rate is not None
    return source.added_rate * source.shift_multipliers[shift_id]


def _ambient_rate(source: FaultSourceConfig, sim_time_hours: float, schedule: FaultSchedule) -> float:
    assert source.humidity_threshold_pct is not None and source.added_rate_above_threshold is not None
    humidity = schedule.humidity_by_hour[int(sim_time_hours)]
    return source.added_rate_above_threshold if humidity > source.humidity_threshold_pct else 0.0
