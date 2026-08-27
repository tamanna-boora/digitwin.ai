"""Trailing-window categorical mix helpers (variant, shift, operator) — same
backward-only window discipline as windowing.py, just for non-numeric fields.
"""

import numpy as np


def fixed_category_fractions_trailing(
    timestamps: np.ndarray,
    categories: np.ndarray,
    bucket_ends: np.ndarray,
    window_seconds: float,
    category_values: list[str],
) -> dict[str, np.ndarray]:
    """Per bucket, fraction of visits belonging to each of a known, fixed set of categories."""
    order = np.argsort(timestamps)
    sorted_ts = timestamps[order]
    sorted_cats = categories[order]

    result = {cv: np.zeros(len(bucket_ends)) for cv in category_values}
    for i, end in enumerate(bucket_ends):
        start = end - window_seconds
        window_cats = sorted_cats[(sorted_ts > start) & (sorted_ts <= end)]
        if window_cats.size == 0:
            continue
        for cv in category_values:
            result[cv][i] = float(np.mean(window_cats == cv))
    return result


def operator_diversity_trailing(
    timestamps: np.ndarray,
    operator_ids: np.ndarray,
    bucket_ends: np.ndarray,
    window_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per bucket: count of distinct operators, and the most-active operator's share."""
    order = np.argsort(timestamps)
    sorted_ts = timestamps[order]
    sorted_ops = operator_ids[order]

    n_distinct = np.zeros(len(bucket_ends))
    dominant_share = np.full(len(bucket_ends), np.nan)
    for i, end in enumerate(bucket_ends):
        start = end - window_seconds
        window_ops = sorted_ops[(sorted_ts > start) & (sorted_ts <= end)]
        if window_ops.size == 0:
            continue
        _, counts = np.unique(window_ops, return_counts=True)
        n_distinct[i] = len(counts)
        dominant_share[i] = float(counts.max() / window_ops.size)
    return n_distinct, dominant_share
