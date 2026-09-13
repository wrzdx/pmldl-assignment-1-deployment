"""Clean raw records, then fit cleaning statistics on training data only."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from iris_ml.common import CLASSES, FEATURES, ROOT, SEED, TARGET, save_json


def prepare(raw: pd.DataFrame):
    required = FEATURES + [TARGET]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = raw[required].copy()
    initial_rows = len(frame)
    frame[FEATURES] = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    frame[FEATURES] = frame[FEATURES].mask(frame[FEATURES] <= 0)
    frame = frame[frame[TARGET].isin(CLASSES)].drop_duplicates()
    if set(frame[TARGET]) != set(CLASSES) or frame[TARGET].value_counts().min() < 5:
        raise ValueError("At least five usable rows per Iris class are required")

    # Splitting first prevents test values from influencing fitted statistics.
    train, test = train_test_split(
        frame, test_size=0.2, stratify=frame[TARGET], random_state=SEED
    )
    train, test = train.copy(), test.copy()
    medians = train[FEATURES].median()
    if medians.isna().any():
        raise ValueError("Training data has an entirely missing feature")
    missing_train = int(train[FEATURES].isna().sum().sum())
    missing_test = int(test[FEATURES].isna().sum().sum())
    train[FEATURES] = train[FEATURES].fillna(medians)
    test[FEATURES] = test[FEATURES].fillna(medians)

    q1, q3 = train[FEATURES].quantile(0.25), train[FEATURES].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    keep = ((train[FEATURES] >= lower) & (train[FEATURES] <= upper)).all(axis=1)
    removed = int((~keep).sum())
    train = train.loc[keep]
    if set(train[TARGET]) != set(CLASSES):
        raise ValueError("Outlier removal eliminated a training class")
    # Keep test outliers: they are part of the honest evaluation distribution.
    report = {
        "raw_rows": initial_rows,
        "invalid_label_or_duplicate_rows_removed": initial_rows - len(frame),
        "training_outliers_removed": removed,
        "train_rows": len(train), "test_rows": len(test),
        "missing_train_values_imputed": missing_train,
        "missing_test_values_imputed": missing_test,
        "train_medians": medians.to_dict(),
        "outlier_lower": lower.to_dict(), "outlier_upper": upper.to_dict(),
        "random_state": SEED,
    }
    return train, test, report


def main():
    raw = pd.read_csv(ROOT / "data/raw/iris.csv")
    train, test, report = prepare(raw)
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    save_json(ROOT / "metrics/data_quality.json", report)
    print(report)


if __name__ == "__main__":
    main()
