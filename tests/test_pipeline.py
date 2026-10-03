from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mlops_pipeline.config import Settings
from mlops_pipeline.exceptions import StageError
from mlops_pipeline.pipelines import run_deployment, run_monitor_and_maybe_retrain, run_training
from mlops_pipeline.serving import create_app


def settings(tmp_path: Path) -> Settings:
    return Settings(store_dir=tmp_path / "store", min_roc_auc=0.8, min_recall=0.6)


def test_training_then_deployment_are_separate(tmp_path: Path):
    cfg = settings(tmp_path)
    train_ctx = run_training(cfg)
    assert train_ctx.values["version"].startswith("v")
    assert "roc_auc" in train_ctx.values["metrics"]
    names = [item.name for item in train_ctx.history]
    assert names == [
        "ingest",
        "validate",
        "preprocess",
        "features",
        "train",
        "evaluate",
        "register",
        "promote",
    ]
    deploy_ctx = run_deployment(cfg)
    assert deploy_ctx.values["served_version"] == train_ctx.values["version"]
    assert (tmp_path / "store" / "serving-bundle" / "model.joblib").exists()
    assert [item.name for item in deploy_ctx.history] == ["package"]


def test_quality_gate(tmp_path: Path):
    cfg = Settings(store_dir=tmp_path / "store", min_roc_auc=1.01, min_recall=0.0)
    try:
        run_training(cfg)
        assert False, "gate should fail"
    except StageError as exc:
        assert "evaluate failed" in str(exc)


def test_retrain_trigger(tmp_path: Path):
    cfg = settings(tmp_path)
    run_training(cfg)
    ctx = run_monitor_and_maybe_retrain(cfg, extra={"observed_psi": 0.4, "observed_error_rate": 0.0})
    assert ctx.values.get("retrain") or ctx.values.get("version")


def test_serving_after_training(tmp_path: Path):
    cfg = settings(tmp_path)
    run_training(cfg)
    client = TestClient(create_app(cfg))
    assert client.get("/ready").json()["ready"] is True
    response = client.post("/score", json={"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    assert response.status_code == 200
    assert 0.0 <= response.json()["score"] <= 1.0
    bad = client.post("/score", json={"features": [1.0]})
    assert bad.status_code == 422


def test_deployment_fails_without_production_alias(tmp_path: Path):
    cfg = settings(tmp_path)
    try:
        run_deployment(cfg)
        assert False
    except StageError as exc:
        assert "package failed" in str(exc)


def test_retrain_not_triggered_on_quiet_window(tmp_path: Path):
    cfg = settings(tmp_path)
    run_training(cfg)
    ctx = run_monitor_and_maybe_retrain(
        cfg, extra={"observed_psi": 0.01, "observed_error_rate": 0.0}
    )
    names = [item.name for item in ctx.history]
    assert "retrain_trigger" in names or "monitor" in names
    assert ctx.values.get("retrain") is not True


def test_serving_503_without_model(tmp_path: Path):
    cfg = settings(tmp_path)
    client = TestClient(create_app(cfg))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["ready"] is False
    denied = client.post("/score", json={"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    assert denied.status_code == 503


def test_validate_rejects_non_finite(tmp_path: Path):
    import numpy as np

    from mlops_pipeline.runtime import PipelineContext, run_stage
    from mlops_pipeline.stages import data
    from mlops_pipeline.store import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    ctx = PipelineContext(run_id="bad")
    ctx.values.update({"store": store, "seed": 1})
    run_stage("ingest", ctx, data.ingest)
    raw = np.load(ctx.values["raw_path"])
    X = raw["X"].copy()
    X[0, 0] = np.nan
    np.savez(ctx.values["raw_path"], X=X, y=raw["y"])
    try:
        run_stage("validate", ctx, data.validate)
        assert False
    except StageError as exc:
        assert "non-finite" in str(exc)


def test_partial_pipeline_does_not_register(tmp_path: Path):
    from mlops_pipeline.pipelines import TRAINING_STAGES, _context
    from mlops_pipeline.runtime import run_stage

    cfg = Settings(store_dir=tmp_path / "store", min_roc_auc=1.01, min_recall=0.0)
    ctx = _context(cfg)
    try:
        for name, fn in TRAINING_STAGES:
            run_stage(name, ctx, fn)
        assert False
    except StageError:
        names = [item.name for item in ctx.history]
        assert "evaluate" in names
        assert "register" not in names
        assert any(not item.ok for item in ctx.history)


def test_training_rerun_is_idempotent(tmp_path: Path):
    cfg = settings(tmp_path)
    first = run_training(cfg)
    second = run_training(cfg)
    assert first.values["version"] != second.values["version"]
    assert first.values["metrics"]["roc_auc"] > 0.8
    assert second.values["metrics"]["roc_auc"] > 0.8


def test_train_without_features_is_partial_failure(tmp_path: Path):
    from mlops_pipeline.runtime import PipelineContext, run_stage
    from mlops_pipeline.stages import model
    from mlops_pipeline.store import ArtifactStore

    ctx = PipelineContext(run_id="partial")
    ctx.values["store"] = ArtifactStore(tmp_path / "store")
    ctx.values["seed"] = 1
    try:
        run_stage("train", ctx, model.train)
        assert False
    except StageError as exc:
        assert "train failed" in str(exc)
        assert ctx.history[0].ok is False
