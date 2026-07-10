from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config/project.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(project_root())
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(cfg["_project_root"]) / path


def class_names(cfg: dict[str, Any]) -> dict[int, str]:
    return {int(k): str(v) for k, v in cfg["classes"].items()}


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

