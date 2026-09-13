"""One-time export; the resulting CSV is included in Git for offline runs."""
from sklearn.datasets import load_iris

from iris_ml.common import CLASSES, FEATURES, ROOT, TARGET


def main():
    iris = load_iris(as_frame=True)
    frame = iris.data.copy()
    frame.columns = FEATURES
    frame[TARGET] = [CLASSES[int(value)] for value in iris.target]
    path = ROOT / "data/raw/iris.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Exported {len(frame)} rows to {path}")


if __name__ == "__main__":
    main()
