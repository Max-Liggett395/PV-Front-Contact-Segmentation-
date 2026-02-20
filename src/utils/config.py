"""Configuration loading utilities."""

import yaml


def load_config(path):
    """Load a YAML config file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_configs(*configs):
    """Merge multiple config dicts. Later configs override earlier ones."""
    merged = {}
    for cfg in configs:
        for key, value in cfg.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged
