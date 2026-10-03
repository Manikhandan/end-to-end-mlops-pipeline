from __future__ import annotations

from uuid import uuid4

from mlops_pipeline.config import Settings, get_settings
from mlops_pipeline.runtime import PipelineContext, run_stage
from mlops_pipeline.stages import data, model, ops
from mlops_pipeline.store import ArtifactStore

TRAINING_STAGES = (
    ("ingest", data.ingest),
    ("validate", data.validate),
    ("preprocess", data.preprocess),
    ("features", data.features),
    ("train", model.train),
    ("evaluate", model.evaluate),
    ("register", model.register),
    ("promote", model.promote),
)

DEPLOYMENT_STAGES = (("package", model.package),)


def _context(settings: Settings, extra: dict | None = None) -> PipelineContext:
    ctx = PipelineContext(run_id=uuid4().hex[:12])
    ctx.values.update(
        {
            "store": ArtifactStore(settings.store_dir),
            "seed": settings.random_seed,
            "min_roc_auc": settings.min_roc_auc,
            "min_recall": settings.min_recall,
            "serving_stage": settings.serving_stage,
            "drift_psi_trigger": settings.drift_psi_trigger,
            "promote_to": "Production",
        }
    )
    if extra:
        ctx.values.update(extra)
    return ctx


def run_training(settings: Settings | None = None, extra: dict | None = None) -> PipelineContext:
    settings = settings or get_settings()
    ctx = _context(settings, extra)
    for name, fn in TRAINING_STAGES:
        run_stage(name, ctx, fn)
    return ctx


def run_deployment(settings: Settings | None = None, extra: dict | None = None) -> PipelineContext:
    settings = settings or get_settings()
    ctx = _context(settings, extra)
    for name, fn in DEPLOYMENT_STAGES:
        run_stage(name, ctx, fn)
    return ctx


def run_monitor_and_maybe_retrain(settings: Settings | None = None, extra: dict | None = None) -> PipelineContext:
    settings = settings or get_settings()
    ctx = _context(settings, extra)
    run_stage("monitor", ctx, ops.monitor)
    try:
        run_stage("retrain_trigger", ctx, ops.retrain_trigger)
    except Exception:
        return ctx
    if ctx.values.get("retrain"):
        return run_training(settings, extra)
    return ctx
