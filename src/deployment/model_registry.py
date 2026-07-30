"""
ModelRegistry — Version-Controlled Model Artefact Management
=============================================================

Tracks every model version with metadata: training date, metrics,
hyperparameters, and file paths. Supports promoting a version to
"production" and loading the best / latest / specific version.

Storage layout
--------------
models/
├── registry.json          ← index of all registered versions
├── forecasting/
│   ├── v1.0_20240101/
│   │   ├── best_gru_model.keras
│   │   └── feature_scaler.pkl
│   └── v1.1_20240215/ ...
└── anomaly/
    ├── v1.0_20240101/
    │   ├── autoencoder.keras
    │   ├── iso_forest.pkl
    │   └── anomaly_scaler.pkl
    └── ...

Usage
-----
>>> registry = ModelRegistry(base_dir="models")
>>> registry.register(
...     version="v1.0",
...     gru_path="models/forecasting/best_gru_model.keras",
...     scaler_path="models/forecasting/feature_scaler.pkl",
...     metrics={"MAE": 28.3, "RMSE": 42.1, "R2": 0.97},
...     params={"units": [128, 64], "dropout": 0.2, "lr": 1e-3},
... )
>>> registry.promote("v1.0")
>>> paths = registry.resolve("production")
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


class ModelRegistry:
    """
    Lightweight file-based model registry.

    Parameters
    ----------
    base_dir : str
        Root directory that contains ``forecasting/`` and ``anomaly/`` sub-dirs.
    """

    REGISTRY_FILE = "registry.json"

    def __init__(self, base_dir: str = "models") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_dir / self.REGISTRY_FILE
        self._index: Dict[str, Any] = self._load_index()

    # ─── Public API ───────────────────────────────────────────────────────────

    def register(
        self,
        version: str,
        gru_path: str,
        scaler_path: str,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        ae_path: Optional[str] = None,
        iso_path: Optional[str] = None,
        ad_scaler_path: Optional[str] = None,
        description: str = "",
    ) -> None:
        """
        Register a new model version in the registry.

        Parameters
        ----------
        version       : Semantic version string, e.g. ``"v1.0"``.
        gru_path      : Absolute or relative path to GRU ``.keras`` file.
        scaler_path   : Path to feature scaler ``.pkl`` file.
        metrics       : Evaluation metrics dict (MAE, RMSE, MAPE, R²).
        params        : Hyperparameter dict used for this run.
        ae_path       : Autoencoder model path (optional).
        iso_path      : Isolation Forest path (optional).
        ad_scaler_path: Anomaly scaler path (optional).
        description   : Free-text description of this version.
        """
        if version in self._index:
            log.warning("Version %s already registered — overwriting.", version)

        entry = {
            "version":        version,
            "registered_at":  datetime.now().isoformat(),
            "description":    description,
            "status":         "registered",   # registered | validated | production
            "metrics":        metrics or {},
            "params":         params or {},
            "paths": {
                "gru_model":      str(gru_path),
                "feature_scaler": str(scaler_path),
                "ae_model":       str(ae_path) if ae_path else None,
                "iso_forest":     str(iso_path) if iso_path else None,
                "ad_scaler":      str(ad_scaler_path) if ad_scaler_path else None,
            },
        }
        self._index[version] = entry
        self._save_index()
        log.info("Registered model version: %s", version)

    def promote(self, version: str, status: str = "production") -> None:
        """
        Promote a version to a given status (``'validated'`` or ``'production'``).
        Only one version can hold ``'production'`` status at a time.
        """
        if version not in self._index:
            raise KeyError(f"Version '{version}' not found in registry.")

        if status == "production":
            # Demote any existing production version
            for v, entry in self._index.items():
                if entry["status"] == "production" and v != version:
                    entry["status"] = "validated"
                    log.info("Demoted %s from production to validated.", v)

        self._index[version]["status"] = status
        self._save_index()
        log.info("Version %s promoted to '%s'.", version, status)

    def resolve(self, version: str = "latest") -> Dict[str, str]:
        """
        Resolve a version string to a dict of artefact file paths.

        Parameters
        ----------
        version : ``"latest"`` | ``"production"`` | ``"best_mae"`` | specific version string

        Returns
        -------
        dict with keys: gru_path, scaler_path, ae_path, iso_path, ad_scaler_path
        """
        if not self._index:
            raise RuntimeError("Model registry is empty. Register a model first.")

        if version == "latest":
            entry = sorted(
                self._index.values(),
                key=lambda e: e["registered_at"],
            )[-1]
        elif version == "production":
            prod = [e for e in self._index.values() if e["status"] == "production"]
            if not prod:
                log.warning("No production version found — falling back to 'latest'.")
                return self.resolve("latest")
            entry = prod[0]
        elif version == "best_mae":
            scored = [e for e in self._index.values() if "MAE" in e.get("metrics", {})]
            if not scored:
                return self.resolve("latest")
            entry = min(scored, key=lambda e: e["metrics"]["MAE"])
        else:
            if version not in self._index:
                raise KeyError(f"Version '{version}' not in registry.")
            entry = self._index[version]

        paths = entry["paths"]
        log.info("Resolved version '%s' → %s  (status: %s)",
                 version, entry["version"], entry["status"])
        return {
            "gru_path":       paths["gru_model"],
            "scaler_path":    paths["feature_scaler"],
            "ae_path":        paths.get("ae_model"),
            "iso_path":       paths.get("iso_forest"),
            "ad_scaler_path": paths.get("ad_scaler"),
        }

    def list_versions(self) -> None:
        """Print a summary table of all registered versions."""
        print(f"\n{'Version':<12} {'Status':<14} {'MAE':>8} {'R2':>7}  {'Registered At'}")
        print("-" * 65)
        for v, e in sorted(self._index.items()):
            mae = e["metrics"].get("MAE", "-")
            r2  = e["metrics"].get("R2",  "-")
            print(f"{v:<12} {e['status']:<14} "
                  f"{mae:>8.2f} {r2:>7.4f}  {e['registered_at'][:19]}"
                  if isinstance(mae, float) else
                  f"{v:<12} {e['status']:<14} {'N/A':>8} {'N/A':>7}  {e['registered_at'][:19]}")
        print()

    def get_metadata(self, version: str) -> Dict[str, Any]:
        """Return the full metadata dict for a specific version."""
        if version not in self._index:
            raise KeyError(f"Version '{version}' not found.")
        return self._index[version]

    def delete(self, version: str, delete_files: bool = False) -> None:
        """
        Remove a version from the registry.

        Parameters
        ----------
        delete_files : If True, also delete the model files on disk.
        """
        if version not in self._index:
            raise KeyError(f"Version '{version}' not found.")
        entry = self._index.pop(version)
        if delete_files:
            for p in entry["paths"].values():
                if p and Path(p).exists():
                    Path(p).unlink()
                    log.info("Deleted file: %s", p)
        self._save_index()
        log.info("Removed version '%s' from registry.", version)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _load_index(self) -> Dict[str, Any]:
        if self._index_path.exists():
            with open(self._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, default=str)
