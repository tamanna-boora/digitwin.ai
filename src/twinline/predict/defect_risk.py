"""defect_risk.py: probability a unit fails end-of-line inspection, from its
journey so far. Time-based split — train on early units, test on late ones —
because tool wear drifts monotonically over the run; a random split would
mix post-drift examples into training and overstate generalization to
conditions the model hasn't actually seen yet. Labels come from defects.csv
(what inspection observed), never ground_truth.csv.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from twinline.data_access import load_defects
from twinline.schemas import DefectRiskConfig, HistGBCConfig

_MODEL_SEED = 42
_CATEGORICAL_COLUMNS = ["variant_id", "shift_id"]
_NON_FEATURE_COLUMNS = {"y", "sequence_number", *_CATEGORICAL_COLUMNS}


@dataclass(frozen=True)
class DefectRiskSplit:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    train_medians: pd.Series


@dataclass(frozen=True)
class DefectRiskModel:
    model: HistGradientBoostingClassifier
    feature_columns: list[str]
    train_medians: pd.Series
    pr_auc_test: float
    n_train: int
    n_test: int
    n_positive_test: int


def build_labeled_dataset(journey_features: pd.DataFrame) -> pd.DataFrame:
    defects = load_defects()
    detected_unit_ids = set(defects.loc[defects["detected"], "unit_id"])
    labeled = journey_features.copy()
    labeled["y"] = labeled.index.to_series().isin(detected_unit_ids).astype(int)
    return labeled


def prepare_features(labeled: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric_columns = [c for c in labeled.columns if c not in _NON_FEATURE_COLUMNS]
    encoded = pd.get_dummies(labeled[_CATEGORICAL_COLUMNS], prefix=_CATEGORICAL_COLUMNS)
    combined = pd.concat([labeled[numeric_columns], encoded, labeled[["y", "sequence_number"]]], axis=1)
    return combined, [*numeric_columns, *encoded.columns]


def time_based_split(labeled: pd.DataFrame, cfg: DefectRiskConfig) -> DefectRiskSplit:
    combined, feature_columns = prepare_features(labeled)
    ordered = combined.sort_values("sequence_number")
    n = len(ordered)
    train_end = int(n * cfg.train_fraction)
    cal_end = int(n * (cfg.train_fraction + cfg.calibration_fraction))

    train = ordered.iloc[:train_end]
    calibration = ordered.iloc[train_end:cal_end]
    test = ordered.iloc[cal_end:]

    # A column with fewer than 2 distinct non-missing values in train (common on a
    # mid-journey checkpoint split, where most stations haven't been visited yet)
    # breaks sklearn's histogram binning outright — see soft_sensor_model.py for the
    # same issue. Drop those before anything downstream tries to fit on them.
    feature_columns = [c for c in feature_columns if train[c].dropna().nunique() >= 2]
    train_medians = train[feature_columns].median(numeric_only=True)

    return DefectRiskSplit(
        train=train, calibration=calibration, test=test, feature_columns=feature_columns, train_medians=train_medians
    )


def train_defect_risk_model(split: DefectRiskSplit, cfg: HistGBCConfig) -> DefectRiskModel:
    x_train = _impute(split.train, split.feature_columns, split.train_medians)
    y_train = split.train["y"].to_numpy(dtype=int)
    sample_weight = _balanced_sample_weights(y_train)

    model = HistGradientBoostingClassifier(
        max_iter=cfg.max_iter,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        random_state=_MODEL_SEED,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)

    x_test = _impute(split.test, split.feature_columns, split.train_medians)
    y_test = split.test["y"].to_numpy(dtype=int)
    proba_test = model.predict_proba(x_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, proba_test)) if y_test.sum() > 0 else float("nan")

    return DefectRiskModel(
        model=model,
        feature_columns=split.feature_columns,
        train_medians=split.train_medians,
        pr_auc_test=pr_auc,
        n_train=len(split.train),
        n_test=len(split.test),
        n_positive_test=int(y_test.sum()),
    )


def predict_defect_risk(risk_model: DefectRiskModel, rows: pd.DataFrame) -> np.ndarray:
    x = build_feature_matrix(risk_model, rows)
    return risk_model.model.predict_proba(x)[:, 1]


def build_feature_matrix(risk_model: DefectRiskModel, rows: pd.DataFrame) -> np.ndarray:
    return _impute(rows, risk_model.feature_columns, risk_model.train_medians)


def _impute(frame: pd.DataFrame, feature_columns: list[str], medians: pd.Series) -> np.ndarray:
    filled = frame.reindex(columns=feature_columns).fillna(medians)
    return filled.to_numpy(dtype=float)


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weight_by_class = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([weight_by_class[v] for v in y])
