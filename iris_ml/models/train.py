import hashlib
import json
from datetime import datetime, timezone

import joblib
import mlflow
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from iris_ml.common import CLASSES, FEATURES, ROOT, SEED, TARGET, save_json


def main():
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("iris-deployment")
    with mlflow.start_run() as run:
        pipeline.fit(train[FEATURES], train[TARGET])
        prediction = pipeline.predict(test[FEATURES])
        metrics = {
            "accuracy": float(accuracy_score(test[TARGET], prediction)),
            "f1_macro": float(f1_score(test[TARGET], prediction, average="macro")),
        }
        metadata = {
            "run_id": run.info.run_id,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "raw_data_sha256": hashlib.sha256((ROOT / "data/raw/iris.csv").read_bytes()).hexdigest(),
            "features": FEATURES, "classes": CLASSES,
            "train_rows": len(train), "test_rows": len(test),
            "metrics": metrics,
        }
        model_path = ROOT / "models/model.joblib"
        model_path.parent.mkdir(exist_ok=True)
        temporary = model_path.with_suffix(".tmp")
        joblib.dump({"pipeline": pipeline, "metadata": metadata}, temporary)
        temporary.replace(model_path)
        save_json(ROOT / "models/metadata.json", metadata)
        save_json(ROOT / "metrics/metrics.json", metrics)
        save_json(ROOT / "metrics/evaluation.json", {
            "class_order": CLASSES,
            "confusion_matrix": confusion_matrix(test[TARGET], prediction, labels=CLASSES).tolist(),
            "classification_report": classification_report(test[TARGET], prediction, output_dict=True),
        })
        mlflow.log_params({"model": "LogisticRegression", "scaler": "StandardScaler",
                           "random_state": SEED, "max_iter": 1000,
                           "train_rows": len(train), "test_rows": len(test)})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(ROOT / "models/metadata.json"), artifact_path="model")
        for name in ("metrics", "evaluation", "data_quality"):
            mlflow.log_artifact(str(ROOT / f"metrics/{name}.json"), artifact_path="reports")
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
