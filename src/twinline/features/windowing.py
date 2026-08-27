"""Pure trailing-window aggregation. Every bucket_end only looks backward —
this is what makes build_station_features safe to point-in-time join later.
"""

import numpy as np
import pandas as pd


def bucket_end_times(min_time_s: float, max_time_s: float, bucket_seconds: float) -> np.ndarray:
    if max_time_s < min_time_s:
        return np.array([], dtype=float)
    n_buckets = max(int(np.ceil((max_time_s - min_time_s) / bucket_seconds)), 1)
    return min_time_s + bucket_seconds + np.arange(n_buckets) * bucket_seconds


def trailing_window_stats(
    timestamps: np.ndarray,
    values: np.ndarray,
    bucket_ends: np.ndarray,
    window_seconds: float,
) -> pd.DataFrame:
    """For each bucket_end, aggregate values with bucket_end - window < timestamp <= bucket_end."""
    order = np.argsort(timestamps)
    sorted_ts = timestamps[order]
    sorted_values = values[order]

    rows: list[dict[str, float | int]] = []
    for end in bucket_ends:
        start = end - window_seconds
        mask = (sorted_ts > start) & (sorted_ts <= end)
        window_values = sorted_values[mask]
        if window_values.size == 0:
            rows.append({"bucket_end_s": end, "mean": np.nan, "std": np.nan, "p95": np.nan, "count": 0})
            continue
        rows.append(
            {
                "bucket_end_s": end,
                "mean": float(np.mean(window_values)),
                "std": float(np.std(window_values, ddof=0)) if window_values.size > 1 else 0.0,
                "p95": float(np.percentile(window_values, 95)),
                "count": int(window_values.size),
            }
        )
    return pd.DataFrame(rows)


def ewma_series(bucket_means: pd.Series, alpha: float) -> pd.Series:
    return bucket_means.ewm(alpha=alpha, adjust=False).mean()


def rolling_slope(series: pd.Series, lookback: int) -> pd.Series:
    def _slope(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        x = np.arange(len(window))
        return float(np.polyfit(x, window, 1)[0])

    return series.rolling(window=lookback, min_periods=lookback).apply(_slope, raw=True)
