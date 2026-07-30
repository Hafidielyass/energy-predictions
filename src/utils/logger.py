"""
Centralised logging configuration.
All modules import `get_logger(__name__)` for consistent log formatting.
"""

import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Return a logger that writes to both console and a rotating log file.

    Parameters
    ----------
    name : str
        Module name (pass ``__name__``).
    log_dir : str
        Directory for log files.

    Returns
    -------
    logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (one file per day)
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now():%Y%m%d}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
