"""Orchestrates one deterministic simulation run from config to typed output."""

import numpy as np

from twinline.schemas import ModelConfig, PlantLineConfig, SimulationOutput
from twinline.sim.line import simulate_line


def run_simulation(plant: PlantLineConfig, model: ModelConfig) -> SimulationOutput:
    rng = np.random.default_rng(model.seed)
    return simulate_line(plant, model, rng)
