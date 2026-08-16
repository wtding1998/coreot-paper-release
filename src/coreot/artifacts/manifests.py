from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Manifest:
    stage: str
    artifacts: dict[str, str]
    metadata: dict[str, Any]


def write_manifest(path: str | Path, manifest: Manifest) -> None:
    payload = {
        "stage": manifest.stage,
        "artifacts": manifest.artifacts,
        "metadata": manifest.metadata,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


def read_manifest(path: str | Path) -> Manifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return Manifest(
        stage=str(payload["stage"]),
        artifacts=dict(payload.get("artifacts", {})),
        metadata=dict(payload.get("metadata", {})),
    )
