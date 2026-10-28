"""Configuration management utilities for loading and validating YAML config files."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file safely.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict containing configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If parsing fails.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


load_config = load_yaml_config
