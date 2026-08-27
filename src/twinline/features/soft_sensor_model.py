"""HistGradientBoostingRegressor quantile trio (0.1/0.5/0.9) per archetype —
an interval, not a point estimate.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from twinline.features.soft_sensor_data import ArchetypeDataset

QUANTILES = (0.1, 0.5, 0.9)
_MODEL_SEED = 42


@dataclass(frozen=True)
class ArchetypeModel:
    archetype_id: str
    feature_columns: list[str]
    quantile_models: dict[float, HistGradientBoostingRegressor]


def fit_archetype_model(archetype_id: str, training: pd.DataFrame, feature_columns: list[str]) -> ArchetypeModel:
    # A column with fewer than 2 distinct non-missing values (common here — thin
    # data, and single-rich-member archetypes leave a whole donor column all-NaN)
    # breaks sklearn's histogram binning outright. Drop those before fitting.
    active_columns = [c for c in feature_columns if training[c].dropna().nunique() >= 2]
    feature_columns = active_columns

    x = training[feature_columns].to_numpy(dtype=float)
    y = training["y"].to_numpy(dtype=float)

    # Training sets here are tiny (tens of bucket-rows) and several target sensors are
    # near-pure noise around a nominal value — an unregularized HGBR happily splits on
    # that noise and reports overconfident (too-narrow) intervals. Heavy regularization
    # (few, shallow trees, large leaves, L2 penalty) keeps it honest: on data with no
    # real signal it should fall back toward the marginal quantiles, not a false pattern.
    min_leaf = max(6, len(training) // 3)
    quantile_models: dict[float, HistGradientBoostingRegressor] = {}
    for q in QUANTILES:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            random_state=_MODEL_SEED,
            max_iter=10,
            max_depth=1,
            min_samples_leaf=min_leaf,
            l2_regularization=2.0,
        )
        model.fit(x, y)
        quantile_models[q] = model

    return ArchetypeModel(archetype_id=archetype_id, feature_columns=feature_columns, quantile_models=quantile_models)


def predict_quantiles(model: ArchetypeModel, rows: pd.DataFrame) -> pd.DataFrame:
    x = rows[model.feature_columns].to_numpy(dtype=float)
    out = pd.DataFrame(index=rows.index)
    for q in QUANTILES:
        out[f"q{int(q * 100)}"] = model.quantile_models[q].predict(x)
    out["lo"] = out["q10"]
    out["value"] = out["q50"]
    out["hi"] = out["q90"]
    # A quantile trio isn't guaranteed monotonic from three independently fit
    # models on thin data — enforce it so lo <= value <= hi always holds.
    out["lo"] = np.minimum(out["lo"], out["value"])
    out["hi"] = np.maximum(out["hi"], out["value"])
    return out[["lo", "value", "hi"]]


def fit_all_archetypes(datasets: dict[str, ArchetypeDataset]) -> dict[str, ArchetypeModel]:
    return {
        archetype_id: fit_archetype_model(archetype_id, ds.training, ds.feature_columns)
        for archetype_id, ds in datasets.items()
    }
