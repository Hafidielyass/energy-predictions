"""
src.utils — Shared Utilities
=============================

Modules
-------
logger             : Centralised logging (console + rotating file handler)
config_loader      : YAML config loader with dot-notation access
experiment_tracker : Lightweight file-based MLOps run tracker
device_config      : TensorFlow device setup, GPU detection, CPU optimisation,
                     warning suppression for Windows TF >= 2.11

Quick usage
-----------
>>> # Always configure TF FIRST before any other TF import
>>> from src.utils.device_config import configure_tf, print_device_report
>>> tf = configure_tf()          # suppresses warnings, tunes threads, returns tf
>>> print_device_report()        # shows device summary

>>> from src.utils.logger import get_logger
>>> from src.utils.config_loader import load_config
>>> from src.utils.experiment_tracker import ExperimentTracker

>>> log = get_logger(__name__)
>>> cfg = load_config("configs/config.yaml")
>>> tracker = ExperimentTracker("my_experiment")
"""

from src.utils.logger import get_logger
from src.utils.config_loader import load_config, Config
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.device_config import configure_tf, print_device_report, get_device_info

__all__ = [
    "get_logger",
    "load_config",
    "Config",
    "ExperimentTracker",
    "configure_tf",
    "print_device_report",
    "get_device_info",
]
