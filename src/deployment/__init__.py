"""
src.deployment — Production Inference & Model Management
=========================================================

Modules
-------
inference_pipeline : End-to-end real-time inference for forecast + anomaly
model_registry     : Version-controlled model artefact registry
batch_scorer       : Offline batch scoring of historical / future windows

Workflow
--------
                  ┌─────────────────────────────────────┐
  Raw sensors ──► │  InferencePipeline.run_forecast()   │ ──► 24h power forecast
                  │  InferencePipeline.run_anomaly()    │ ──► severity label + score
                  └─────────────────────────────────────┘
                            ▲
                     ModelRegistry.load_best()
                     (resolves to latest validated version)

Key classes
-----------
InferencePipeline  — single entry-point for both forecast and anomaly tasks
ModelRegistry      — register, version, promote, and load model artefacts
BatchScorer        — score an entire Parquet file and save results

Usage
-----
>>> from src.deployment import InferencePipeline, ModelRegistry
>>> registry = ModelRegistry()
>>> pipeline = InferencePipeline.from_registry(registry)
>>> forecast  = pipeline.run_forecast(recent_df)
>>> anomaly   = pipeline.run_anomaly(daily_df)
"""

from src.deployment.inference_pipeline import InferencePipeline
from src.deployment.model_registry import ModelRegistry
from src.deployment.batch_scorer import BatchScorer

__all__ = ["InferencePipeline", "ModelRegistry", "BatchScorer"]
