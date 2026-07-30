"""
YAML configuration loader with dot-notation access.
"""

import yaml
from pathlib import Path


class Config(dict):
    """Dict subclass that supports attribute-style access."""

    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict):
                return Config(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str = "configs/config.yaml") -> Config:
    """
    Load the YAML config file and return a Config object.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    Config
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(raw)
