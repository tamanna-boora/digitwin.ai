"""Held-out validation: hold out some rich stations' LATER buckets in time,
fit fresh quantile models on the earlier ones, and check what fraction of
true values land inside the predicted 10-90 interval. Distinct from the
leave-one-out donor construction in soft_sensor_data.py — that prevents a
station's own signal leaking into its own features, but a model trained on
the full pooled table would still have seen a rich station's own bucket-rows
during fitting. This does a genuine chronological train/test split so the
number reported is an honest out-of-sample coverage check, not in-sample fit.
"""

from dataclasses import dataclass

from twinline.features.soft_sensor_data import ArchetypeDataset
from twinline.features.soft_sensor_model import fit_archetype_model, predict_quantiles

TRAIN_FRACTION = 0.7


@dataclass(frozen=True)
class ArchetypeValidation:
    archetype_id: str
    n_train: int
    n_test: int
    coverage_fraction: float


def validate_soft_sensors(datasets: dict[str, ArchetypeDataset]) -> list[ArchetypeValidation]:
    results = []
    for archetype_id, dataset in datasets.items():
        results.append(_validate_one_archetype(archetype_id, dataset))
    return results


def _validate_one_archetype(archetype_id: str, dataset: ArchetypeDataset) -> ArchetypeValidation:
    ordered = dataset.training.sort_values("bucket_end_s")
    split_index = max(int(len(ordered) * TRAIN_FRACTION), 1)
    train_rows = ordered.iloc[:split_index]
    test_rows = ordered.iloc[split_index:]

    if train_rows.empty or test_rows.empty:
        return ArchetypeValidation(archetype_id, len(train_rows), len(test_rows), float("nan"))

    model = fit_archetype_model(archetype_id, train_rows, dataset.feature_columns)
    pred = predict_quantiles(model, test_rows)

    inside = (test_rows["y"].to_numpy() >= pred["lo"].to_numpy()) & (test_rows["y"].to_numpy() <= pred["hi"].to_numpy())
    coverage = float(inside.mean())

    return ArchetypeValidation(archetype_id, len(train_rows), len(test_rows), coverage)


def print_validation_report(results: list[ArchetypeValidation]) -> None:
    total_test = sum(r.n_test for r in results)
    total_inside = sum(r.coverage_fraction * r.n_test for r in results if r.n_test > 0)

    print(f"{'archetype':<16}{'n_train':>9}{'n_test':>9}{'coverage':>12}")
    for r in results:
        cov_str = f"{r.coverage_fraction:.1%}" if r.n_test > 0 else "n/a"
        print(f"{r.archetype_id:<16}{r.n_train:>9}{r.n_test:>9}{cov_str:>12}")

    overall = total_inside / total_test if total_test > 0 else float("nan")
    print(f"\noverall 10-90 interval coverage: {overall:.1%} (target: near 80%, n={total_test})")
