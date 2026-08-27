"""Pure control-chart rule functions, each returning a boolean out-of-control
mask aligned to the input series. No station/sensor identity here — that's
detect/spc.py's job, this module only knows numbers.
"""

import numpy as np
import pandas as pd

from twinline.schemas import SPCConfig


def ewma_mask(values: np.ndarray, alpha: float, center: float, sigma: float, control_limit_l: float) -> np.ndarray:
    ewma = pd.Series(values).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    n = np.arange(1, len(values) + 1)
    limit = control_limit_l * sigma * np.sqrt((alpha / (2 - alpha)) * (1 - (1 - alpha) ** (2 * n)))
    return np.abs(ewma - center) > limit


def cusum_mask(values: np.ndarray, center: float, sigma: float, k_sigma: float, h_sigma: float) -> np.ndarray:
    k = k_sigma * sigma
    h = h_sigma * sigma
    upper = np.zeros(len(values))
    lower = np.zeros(len(values))
    for i, v in enumerate(values):
        prev_upper = upper[i - 1] if i > 0 else 0.0
        prev_lower = lower[i - 1] if i > 0 else 0.0
        upper[i] = max(0.0, prev_upper + (v - center) - k)
        lower[i] = min(0.0, prev_lower + (v - center) + k)
    return (upper > h) | (lower < -h)


def rule1_beyond_3sigma(values: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.abs(values - center) > 3.0 * sigma


def rule2_two_of_three_beyond_2sigma(values: np.ndarray, center: float, sigma: float, cfg: SPCConfig) -> np.ndarray:
    beyond = np.abs(values - center) > cfg.rule2_sigma * sigma
    same_side = np.sign(values - center)
    return _rolling_same_side_hits(beyond, same_side, cfg.rule2_window, cfg.rule2_hits)


def rule3_four_of_five_beyond_1sigma(values: np.ndarray, center: float, sigma: float, cfg: SPCConfig) -> np.ndarray:
    beyond = np.abs(values - center) > cfg.rule3_sigma * sigma
    same_side = np.sign(values - center)
    return _rolling_same_side_hits(beyond, same_side, cfg.rule3_window, cfg.rule3_hits)


def rule4_run_same_side(values: np.ndarray, center: float, run_length: int) -> np.ndarray:
    side = np.sign(values - center)
    mask = np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        start = max(0, i - run_length + 1)
        window = side[start : i + 1]
        if len(window) == run_length and window[0] != 0 and np.all(window == window[0]):
            mask[i] = True
    return mask


def _rolling_same_side_hits(beyond: np.ndarray, same_side: np.ndarray, window: int, hits: int) -> np.ndarray:
    mask = np.zeros(len(beyond), dtype=bool)
    for i in range(len(beyond)):
        start = max(0, i - window + 1)
        w_beyond = beyond[start : i + 1]
        w_side = same_side[start : i + 1]
        if len(w_beyond) < window:
            continue
        for target_side in (1, -1):
            hit_count = np.sum(w_beyond & (w_side == target_side))
            if hit_count >= hits:
                mask[i] = True
    return mask
