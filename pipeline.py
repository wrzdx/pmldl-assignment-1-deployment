"""Run DVC processing/training and Docker deployment once, or every 300s."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from filelock import FileLock, Timeout
import requests

ROOT = Path(__file__).resolve().parent
COMPOSE = ROOT / "iris_ml/deployment/docker-compose.yml"
LOG = logging.getLogger("pipeline")


def command(args):
    LOG.info("Executing: %s", " ".join(args))
    env = os.environ.copy()
    # DVC's `python` commands must use the same environment as this runner.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def smoke_check():
    api = f"http://127.0.0.1:{os.getenv('API_PORT', '8000')}"
    app = f"http://127.0.0.1:{os.getenv('APP_PORT', '8501')}"
    metadata = json.loads((ROOT / "models/metadata.json").read_text())
    health = requests.get(f"{api}/health", timeout=15)
    health.raise_for_status()
    if health.json()["run_id"] != metadata["run_id"]:
        raise RuntimeError("API is serving a stale model")
    response = requests.post(f"{api}/predict", json={
        "sepal_length": 5.1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2,
    }, timeout=15)
    response.raise_for_status()
    prediction = response.json()
    if prediction["species"] != "setosa" or prediction["run_id"] != metadata["run_id"]:
        raise RuntimeError(f"Prediction smoke check failed: {prediction}")
    requests.get(f"{app}/_stcore/health", timeout=15).raise_for_status()
    # Verify the network path used by the Streamlit container as well.
    command(["docker", "compose", "-f", str(COMPOSE), "exec", "-T", "app",
             "python", "-c",
             "import requests; r=requests.post('http://api:8000/predict', "
             "json={'sepal_length':5.1,'sepal_width':3.5,'petal_length':1.4,'petal_width':0.2}, "
             "timeout=10); r.raise_for_status(); assert r.json()['species']=='setosa'"])
    return metadata["run_id"]


def run_once():
    started = time.monotonic()
    record = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "failed"}
    try:
        LOG.info("Stages 1–2: data preparation and model engineering")
        command([sys.executable, "-m", "dvc", "repro", "--force"])
        LOG.info("Stage 3: build and deploy both containers")
        command(["docker", "compose", "-f", str(COMPOSE), "up", "--build",
                 "--detach", "--force-recreate", "--wait", "--wait-timeout", "120"])
        record["run_id"] = smoke_check()
        record["status"] = "success"
        LOG.info("All three stages passed; model run_id=%s", record["run_id"])
    except Exception as exc:
        record["error"] = str(exc)
        raise
    finally:
        record["duration_seconds"] = round(time.monotonic() - started, 3)
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        with (ROOT / "logs/runs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")


def next_delay(started, finished, interval):
    """Skip elapsed slots; never overlap runs or replay a backlog."""
    slots = max(1, math.floor((finished - started) / interval) + 1)
    return max(0.0, started + slots * interval - finished)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run all stages once, then exit")
    parser.add_argument("--interval", type=int, default=300, help="Start interval in seconds (default 300)")
    parser.add_argument("--max-runs", type=int, default=0, help="Stop after N runs; 0 means forever")
    args = parser.parse_args()
    if args.interval < 1 or args.max_runs < 0:
        parser.error("interval must be positive and max-runs nonnegative")
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "logs/pipeline.log", encoding="utf-8")])
    try:
        with FileLock(ROOT / "logs/pipeline.lock", timeout=0):
            count = 0
            while True:
                started = time.monotonic()
                try:
                    run_once()
                except Exception:
                    LOG.exception("Pipeline failed; deployment was not confirmed")
                    if args.once or args.max_runs:
                        return 1
                count += 1
                if args.once or (args.max_runs and count >= args.max_runs):
                    return 0
                delay = next_delay(started, time.monotonic(), args.interval)
                LOG.info("Next complete run in %.1f seconds; Ctrl+C stops the scheduler", delay)
                time.sleep(delay)
    except Timeout:
        LOG.error("Another scheduler/run already holds the pipeline lock")
        return 2
    except KeyboardInterrupt:
        LOG.info("Scheduler stopped; Docker containers continue running")
        return 0


if __name__ == "__main__":
    sys.exit(main())
