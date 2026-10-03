from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_pipeline.exceptions import GateError
from mlops_pipeline.runtime import PipelineContext
from mlops_pipeline.store import ArtifactStore


def train(ctx: PipelineContext) -> dict:
    store: ArtifactStore = ctx.require("store")
    seed: int = ctx.require("seed")
    data = store.load_npz(ctx.require("feature_path"))
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed)),
        ]
    )
    model.fit(data["X_train"], data["y_train"])
    path = store.run_dir(ctx.run_id) / "model.joblib"
    store.save_model(path, model)
    return {"model_path": str(path)}


def evaluate(ctx: PipelineContext) -> dict:
    store: ArtifactStore = ctx.require("store")
    data = store.load_npz(ctx.require("feature_path"))
    model = store.load_model(ctx.require("model_path"))
    scores = model.predict_proba(data["X_valid"])[:, 1]
    labels = (scores >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(data["y_valid"], scores)),
        "recall": float(recall_score(data["y_valid"], labels, zero_division=0)),
        "n_valid": float(len(data["y_valid"])),
    }
    min_auc = ctx.require("min_roc_auc")
    min_recall = ctx.require("min_recall")
    if metrics["roc_auc"] < min_auc:
        raise GateError(f"roc_auc {metrics['roc_auc']:.3f} < {min_auc}")
    if metrics["recall"] < min_recall:
        raise GateError(f"recall {metrics['recall']:.3f} < {min_recall}")
    report = store.run_dir(ctx.run_id) / "metrics.json"
    store.write_json(report, metrics)
    return {"metrics": metrics, "metrics_path": str(report)}


def register(ctx: PipelineContext) -> dict:
    store: ArtifactStore = ctx.require("store")
    version = store.register(ctx.run_id, ctx.require("metrics"), ctx.require("model_path"))
    return {"version": version}


def promote(ctx: PipelineContext) -> dict:
    store: ArtifactStore = ctx.require("store")
    stage = ctx.values.get("promote_to", "Staging")
    record = store.promote(ctx.require("version"), stage)
    return {"promoted": record}


def package(ctx: PipelineContext) -> dict:
    """Deployment pipeline: freeze the Production alias into a serving bundle."""
    store: ArtifactStore = ctx.require("store")
    record = store.resolve(ctx.require("serving_stage"))
    bundle = store.root / "serving-bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    model_src = record["path"]
    dest = bundle / "model.joblib"
    dest.write_bytes(open(model_src, "rb").read())
    store.write_json(bundle / "bundle.json", {"version": record["version"], "metrics": record["metrics"]})
    return {"bundle_dir": str(bundle), "served_version": record["version"]}
