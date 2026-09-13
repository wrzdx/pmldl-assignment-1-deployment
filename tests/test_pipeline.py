import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from streamlit.testing.v1 import AppTest

from iris_ml.common import CLASSES, FEATURES, ROOT, TARGET
from iris_ml.datasets.prepare import prepare
from iris_ml.deployment.api.main import app
from iris_ml.models import train as train_module
import pipeline as runner


@pytest.fixture
def raw():
    iris = load_iris(as_frame=True)
    frame = iris.data.copy()
    frame.columns = FEATURES
    frame[TARGET] = [CLASSES[int(x)] for x in iris.target]
    return frame


def test_cleaning_imputes_and_removes_training_outliers(raw):
    baseline, _, _ = prepare(raw)
    raw.loc[baseline.index[0], FEATURES[0]] = 10000
    raw.loc[baseline.index[1], FEATURES[1]] = np.nan
    train, test, report = prepare(raw)
    assert report["training_outliers_removed"] >= 1
    assert report["missing_train_values_imputed"] >= 1
    assert not train.isna().any().any()
    assert not test.isna().any().any()
    assert train[FEATURES[0]].max() < 10000
    assert set(train.index).isdisjoint(test.index)
    assert set(train[TARGET]) == set(CLASSES)


def test_test_data_does_not_influence_training_statistics(raw):
    before_train, before_test, before = prepare(raw)
    raw.loc[before_test.index[0], FEATURES[0]] = 10000
    after_train, after_test, after = prepare(raw)
    pd.testing.assert_frame_equal(before_train, after_train)
    assert before["train_medians"] == after["train_medians"]
    assert before["outlier_upper"] == after["outlier_upper"]
    assert len(after_test) == len(before_test)
    assert after_test[FEATURES[0]].max() == 10000


def test_invalid_schema_is_rejected(raw):
    with pytest.raises(ValueError, match="Missing columns"):
        prepare(raw.drop(columns=FEATURES[0]))


@pytest.fixture
def trained(raw, tmp_path, monkeypatch):
    train, test, report = prepare(raw)
    for folder in ("data/raw", "data/processed", "metrics"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    raw.to_csv(tmp_path / "data/raw/iris.csv", index=False)
    train.to_csv(tmp_path / "data/processed/train.csv", index=False)
    test.to_csv(tmp_path / "data/processed/test.csv", index=False)
    (tmp_path / "metrics/data_quality.json").write_text(json.dumps(report))
    monkeypatch.setattr(train_module, "ROOT", tmp_path)
    train_module.main()
    return tmp_path


def test_model_round_trip_and_mlflow_logging(trained):
    bundle = joblib.load(trained / "models/model.joblib")
    row = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=FEATURES)
    assert bundle["pipeline"].predict(row)[0] == "setosa"
    metrics = json.loads((trained / "metrics/metrics.json").read_text())
    assert metrics["accuracy"] >= 0.85
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=(trained / "mlruns").as_uri())
    run_id = bundle["metadata"]["run_id"]
    assert client.get_run(run_id).data.metrics["accuracy"] == metrics["accuracy"]
    assert any(x.path == "model/model.joblib" for x in client.list_artifacts(run_id, "model"))


def test_api_prediction_and_validation(trained, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained / "models/model.joblib"))
    payload = dict(zip(FEATURES, [5.1, 3.5, 1.4, 0.2]))
    with TestClient(app) as client:
        result = client.post("/predict", json=payload)
        assert result.status_code == 200
        assert result.json()["species"] == "setosa"
        assert sum(result.json()["probabilities"].values()) == pytest.approx(1)
        assert client.get("/health").json()["run_id"] == result.json()["run_id"]
        assert client.get("/model").json()["features"] == FEATURES
        for bad in ({}, {**payload, "petal_width": -1},
                    {**payload, "extra": 1}, {**payload, "sepal_width": "bad"},
                    {**payload, "petal_width": 100}):
            assert client.post("/predict", json=bad).status_code == 422


def test_streamlit_form_displays_api_result(monkeypatch):
    import requests
    calls = []

    def post(url, json, timeout):
        calls.append((url, json, timeout))
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"species":"setosa","probabilities":{"setosa":0.98,"versicolor":0.01,"virginica":0.01},"run_id":"test-model"}'
        return response

    monkeypatch.setattr(requests, "post", post)
    view = AppTest.from_file(str(ROOT / "iris_ml/deployment/app/main.py")).run()
    assert len(view.number_input) == 4
    view.button[0].click().run()
    assert not view.exception
    assert "setosa" in view.success[0].value
    assert calls[0][1] == dict(zip(FEATURES, [5.1, 3.5, 1.4, 0.2]))


def test_streamlit_handles_unavailable_api(monkeypatch):
    import requests

    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "post", unavailable)
    view = AppTest.from_file(str(ROOT / "iris_ml/deployment/app/main.py")).run()
    view.button[0].click().run()
    assert not view.exception
    assert len(view.error) == 1


@pytest.mark.parametrize("finished,expected", [(20, 280), (300, 300), (650, 250)])
def test_scheduler_skips_missed_slots(finished, expected):
    assert runner.next_delay(0, finished, 300) == expected


def test_failed_training_never_deploys(tmp_path, monkeypatch):
    import subprocess
    calls = []
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    def fail(args):
        calls.append(args)
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(runner, "command", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_once()
    assert len(calls) == 1
    assert "dvc" in calls[0]
    record = json.loads((tmp_path / "logs/runs.jsonl").read_text())
    assert record["status"] == "failed"
