from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class ArtifactStore:
    """Layout: var/store/runs/<id>/ and var/store/registry/."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs = root / "runs"
        self.registry = root / "registry"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.registry.mkdir(parents=True, exist_ok=True)
        self._index = self.registry / "index.json"
        if not self._index.exists():
            self._index.write_text(json.dumps({"aliases": {}, "versions": []}, indent=2))

    def run_dir(self, run_id: str) -> Path:
        path = self.runs / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text())

    def save_npz(self, path: Path, **arrays: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    def load_npz(self, path: Path) -> dict[str, np.ndarray]:
        data = np.load(path, allow_pickle=False)
        return {key: data[key] for key in data.files}

    def save_model(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)

    def load_model(self, path: Path) -> Any:
        return joblib.load(path)

    def register(self, run_id: str, metrics: dict[str, float], model_path: Path) -> str:
        index = self.read_json(self._index)
        version = f"v{len(index['versions']) + 1:03d}"
        dest = self.registry / version
        dest.mkdir(parents=True, exist_ok=False)
        artifact = dest / "model.joblib"
        artifact.write_bytes(Path(model_path).read_bytes())
        record = {
            "version": version,
            "run_id": run_id,
            "metrics": metrics,
            "stage": "None",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "path": str(artifact),
        }
        self.write_json(dest / "metadata.json", record)
        index["versions"].append(record)
        self.write_json(self._index, index)
        return version

    def promote(self, version: str, stage: str) -> dict[str, Any]:
        index = self.read_json(self._index)
        found = None
        for item in index["versions"]:
            if item["version"] == version:
                found = item
            elif item.get("stage") == stage:
                item["stage"] = "Archived"
        if found is None:
            raise FileNotFoundError(version)
        found["stage"] = stage
        index["aliases"][stage] = version
        self.write_json(self._index, index)
        return found

    def resolve(self, stage: str) -> dict[str, Any]:
        index = self.read_json(self._index)
        version = index["aliases"].get(stage)
        if not version:
            raise FileNotFoundError(stage)
        for item in index["versions"]:
            if item["version"] == version:
                return item
        raise FileNotFoundError(version)
