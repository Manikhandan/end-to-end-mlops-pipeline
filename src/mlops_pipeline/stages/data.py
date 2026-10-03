from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlops_pipeline.exceptions import GateError, StageError
from mlops_pipeline.features import enrich
from mlops_pipeline.runtime import PipelineContext
from mlops_pipeline.store import ArtifactStore


def ingest(ctx: PipelineContext) -> dict:
    store: ArtifactStore = ctx.require("store")
    seed: int = ctx.require("seed")
    X, y = make_classification(
        n_samples=1600,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        class_sep=1.2,
        random_state=seed,
    )
    path = store.run_dir(ctx.run_id) / "raw.npz"
    store.save_npz(path, X=X, y=y)
    return {"raw_path": str(path), "n_rows": int(len(y))}


def validate(ctx: PipelineContext) -> dict:
    store: ArtifactStore = ctx.require("store")
    data = store.load_npz(ctx.require("raw_path"))
    X, y = data["X"], data["y"]
    if not np.isfinite(X).all():
        raise StageError("non-finite features")
    if len(np.unique(y)) != 2:
        raise StageError("expected binary labels")
    pos = float(y.mean())
    if pos < 0.2 or pos > 0.8:
        raise GateError(f"class balance {pos:.2f} outside 0.2-0.8")
    return {"validated": True, "positive_rate": pos}


def preprocess(ctx: PipelineContext) -> dict:
    store: ArtifactStore = ctx.require("store")
    seed: int = ctx.require("seed")
    data = store.load_npz(ctx.require("raw_path"))
    X_train, X_valid, y_train, y_valid = train_test_split(
        data["X"], data["y"], test_size=0.25, stratify=data["y"], random_state=seed
    )
    path = store.run_dir(ctx.run_id) / "split.npz"
    store.save_npz(path, X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)
    return {"split_path": str(path)}


def features(ctx: PipelineContext) -> dict:
    """Identity features plus a cheap interaction so the stage is a real boundary."""
    store: ArtifactStore = ctx.require("store")
    data = store.load_npz(ctx.require("split_path"))
    path = store.run_dir(ctx.run_id) / "features.npz"
    store.save_npz(
        path,
        X_train=enrich(data["X_train"]),
        X_valid=enrich(data["X_valid"]),
        y_train=data["y_train"],
        y_valid=data["y_valid"],
    )
    return {"feature_path": str(path), "n_features": 7}
