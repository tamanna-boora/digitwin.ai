"""YAML config loading. This is the one sanctioned I/O boundary inside src/."""

from pathlib import Path

import yaml
from pydantic import BaseModel

from twinline.schemas import DetectConfig, FeaturesConfig, ModelConfig, PlantLineConfig, SoftSensorsConfig

DEFAULT_PLANT_CONFIG_PATH = Path("configs/plant_line_a.yaml")
DEFAULT_MODEL_CONFIG_PATH = Path("configs/model.yaml")
DEFAULT_FEATURES_CONFIG_PATH = Path("configs/features.yaml")
DEFAULT_SOFT_SENSORS_CONFIG_PATH = Path("configs/soft_sensors.yaml")
DEFAULT_DETECT_CONFIG_PATH = Path("configs/detect.yaml")


class AppConfig(BaseModel):
    plant: PlantLineConfig
    model: ModelConfig


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_plant_config(path: Path = DEFAULT_PLANT_CONFIG_PATH) -> PlantLineConfig:
    return PlantLineConfig.model_validate(_read_yaml(path))


def load_model_config(path: Path = DEFAULT_MODEL_CONFIG_PATH) -> ModelConfig:
    return ModelConfig.model_validate(_read_yaml(path))


def load_features_config(path: Path = DEFAULT_FEATURES_CONFIG_PATH) -> FeaturesConfig:
    return FeaturesConfig.model_validate(_read_yaml(path))


def load_detect_config(path: Path = DEFAULT_DETECT_CONFIG_PATH) -> DetectConfig:
    return DetectConfig.model_validate(_read_yaml(path))


def load_soft_sensors_config(
    path: Path = DEFAULT_SOFT_SENSORS_CONFIG_PATH, plant: PlantLineConfig | None = None
) -> SoftSensorsConfig:
    cfg = SoftSensorsConfig.model_validate(_read_yaml(path))
    if plant is not None:
        _cross_validate_soft_sensors(cfg, plant)
    return cfg


def _cross_validate_soft_sensors(cfg: SoftSensorsConfig, plant: PlantLineConfig) -> None:
    station_ids = {s.id for s in plant.stations}
    for archetype in cfg.archetypes:
        for station_id in [*archetype.rich_members, *archetype.blind_members]:
            if station_id not in station_ids:
                raise ValueError(f"archetype {archetype.id} references unknown station {station_id}")
        for station_id in archetype.rich_members:
            station = plant.station_by_id(station_id)
            if archetype.target_sensor not in station.sensors:
                raise ValueError(
                    f"archetype {archetype.id}: rich member {station_id} does not measure "
                    f"target sensor {archetype.target_sensor}"
                )
        for station_id in archetype.blind_members:
            station = plant.station_by_id(station_id)
            if archetype.target_sensor in station.sensors:
                raise ValueError(
                    f"archetype {archetype.id}: blind member {station_id} already measures "
                    f"{archetype.target_sensor} directly — it isn't blind for this sensor"
                )


def load_app_config(
    plant_path: Path = DEFAULT_PLANT_CONFIG_PATH,
    model_path: Path = DEFAULT_MODEL_CONFIG_PATH,
) -> AppConfig:
    plant = load_plant_config(plant_path)
    model = load_model_config(model_path)
    _cross_validate(plant, model)
    return AppConfig(plant=plant, model=model)


def _cross_validate(plant: PlantLineConfig, model: ModelConfig) -> None:
    station_ids = {s.id for s in plant.stations}
    defect_capable_ids = {s.id for s in plant.stations if s.can_cause_defect}

    for station in plant.stations:
        for sensor_name in station.sensors:
            if sensor_name != "cycle_time_s" and sensor_name not in model.sensor_specs:
                raise ValueError(f"station {station.id} uses sensor {sensor_name}, missing from model.sensor_specs")

    for source in model.fault_sources:
        gate = plant.gate_for_defect_type(source.defect_type)
        gate_sequence = plant.station_by_id(gate.station_id).sequence
        for station_id in source.station_ids:
            if station_id not in station_ids:
                raise ValueError(f"fault source {source.id} references unknown station {station_id}")
            if station_id not in defect_capable_ids:
                raise ValueError(
                    f"fault source {source.id} targets station {station_id}, "
                    "which is not marked can_cause_defect"
                )
            if plant.station_by_id(station_id).sequence >= gate_sequence:
                raise ValueError(
                    f"fault source {source.id} at {station_id} sequences at or after its "
                    f"detection gate {gate.station_id} — the defect could never be caught"
                )
