"""Minimal filesystem-backed run registry for offline experiments."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> str:
    return str(value)


def _safe_run_id(run_id: str | None) -> str:
    raw = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw)).strip("-._")
    return safe or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class OfflineRegistry:
    """Append-only local registry for one experiment run."""

    def __init__(
        self,
        root: str | Path,
        run_id: str | None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.run_id = _safe_run_id(run_id)
        self.run_dir = self.root / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.write_run_metadata(config=config or {}, metadata=metadata or {})

    def write_run_metadata(self, config: dict[str, Any], metadata: dict[str, Any]) -> Path:
        payload = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "metadata": metadata,
        }
        return self.write_json("run.json", payload)

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        return path

    def log_metrics(self, metrics: dict[str, Any]) -> Path:
        payload = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, default=_json_default)
            handle.write("\n")
        return self.metrics_path

    def copy_artifact(self, path: str | Path, name: str | None = None) -> Path:
        source = Path(path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        destination = self.artifacts_dir / (name or source.name)
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        return destination


def init_offline_registry(
    root: str | Path | None,
    run_id: str | None,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> OfflineRegistry | None:
    if not root:
        return None
    return OfflineRegistry(root=root, run_id=run_id, config=config, metadata=metadata)
