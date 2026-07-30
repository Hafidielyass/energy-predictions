"""
Lightweight experiment tracker that logs metrics, parameters, and
artefact paths to JSON files — no external MLflow server required.
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


class ExperimentTracker:
    """
    Simple file-based experiment tracker.

    Each *run* produces one JSON file under ``experiments/<run_id>.json``.
    A ``runs_index.json`` keeps a summary of every run for quick lookup.

    Usage
    -----
    >>> tracker = ExperimentTracker(experiment_name="gru_forecast_v1")
    >>> tracker.log_params({"units": 128, "dropout": 0.1, "lr": 1e-3})
    >>> tracker.log_metrics({"val_loss": 0.023, "MAE": 12.4})
    >>> tracker.log_artifact("models/forecasting/best_gru_model.keras")
    >>> tracker.finish()
    """

    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "experiments",
    ) -> None:
        self.experiment_name = experiment_name
        self.run_id = f"{experiment_name}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._data: Dict[str, Any] = {
            "run_id": self.run_id,
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_s": None,
            "status": "running",
            "params": {},
            "metrics": {},
            "artifacts": [],
            "tags": {},
        }
        self._start_wall = time.time()
        log.info("Experiment run started: %s", self.run_id)

    # ─── Public API ──────────────────────────────────────────────────────

    def log_params(self, params: Dict[str, Any]) -> None:
        self._data["params"].update(params)
        log.info("Params logged: %s", params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            if k not in self._data["metrics"]:
                self._data["metrics"][k] = []
            entry = {"value": float(v)}
            if step is not None:
                entry["step"] = step
            self._data["metrics"][k].append(entry)
        log.info("Metrics logged: %s", metrics)

    def log_artifact(self, path: str) -> None:
        self._data["artifacts"].append(str(path))
        log.info("Artifact registered: %s", path)

    def set_tag(self, key: str, value: str) -> None:
        self._data["tags"][key] = value

    def finish(self, status: str = "completed") -> None:
        self._data["status"] = status
        self._data["end_time"] = datetime.now().isoformat()
        self._data["duration_s"] = round(time.time() - self._start_wall, 2)
        self._save()
        self._update_index()
        log.info("Experiment run finished: %s  [%s]  %.1f s",
                 self.run_id, status, self._data["duration_s"])

    # ─── Internal ────────────────────────────────────────────────────────

    def _save(self) -> None:
        out = self.run_dir / "run_meta.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)

    def _update_index(self) -> None:
        index_path = self.base_dir / "runs_index.json"
        index: list = []
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)

        summary = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "start_time": self._data["start_time"],
            "duration_s": self._data["duration_s"],
            "status": self._data["status"],
            "final_metrics": {
                k: v[-1]["value"] for k, v in self._data["metrics"].items() if v
            },
        }
        index.append(summary)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, default=str)
