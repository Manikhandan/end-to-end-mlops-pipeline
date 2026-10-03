from __future__ import annotations

import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from mlops_pipeline.config import Settings, get_settings
from mlops_pipeline.features import enrich
from mlops_pipeline.store import ArtifactStore


class ScoreIn(BaseModel):
    features: list[float] = Field(..., min_length=6, max_length=6)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = ArtifactStore(settings.store_dir)
    app = FastAPI(title="mlops-pipeline-serving")
    app.state.settings = settings
    app.state.store = store
    app.state.model = None
    app.state.record = None
    try:
        record = store.resolve(settings.serving_stage)
        app.state.model = store.load_model(record["path"])
        app.state.record = record
    except FileNotFoundError:
        pass

    @app.middleware("http")
    async def rid(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if app.state.model is None:
            return {"ready": False, "reason": "bundle_missing"}
        return {"ready": True, "version": app.state.record["version"]}

    @app.post("/score")
    def score(body: ScoreIn):
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        X = enrich(np.asarray(body.features, dtype=float).reshape(1, -1))
        proba = float(app.state.model.predict_proba(X)[0, 1])
        return {"score": proba, "label": int(proba >= 0.5), "version": app.state.record["version"]}

    return app


app = create_app()
