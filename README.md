# End-to-End MLOps Pipeline

This is the complete ML lifecycle, not only a deploy script.

The **ML development graph** is ingest → validate → preprocess → features → train → evaluate. The **MLOps automation graph** is register → promote, then a separate **deployment graph** that packages the Production alias into a serving bundle. A monitor stage can raise a retrain trigger when PSI or error rate crosses a threshold. Airflow is an optional adapter; pytest runs the same callables.

Original public repository. Synthetic data. No live scheduler is claimed.

## Why two pipelines

Training is a batch job with quality gates. Deployment is a freeze of a specific promoted version. Mixing them hides the moment a model becomes the thing you actually serve. Airflow is optional (`dags/mlops_dag.py`); the graphs run with `python scripts/run_training.py` and `python scripts/run_deployment.py`.

```
ingest → validate → preprocess → features → train → evaluate ─┐
                                                              ├ register → promote (Production)
monitor (PSI / error rate) ─ retrain trigger ─────────────────┘

Production alias → package → serving bundle → FastAPI /score
```

## Local setup

```bash
pip install -e ".[dev]"
cp .env.example .env
python scripts/run_training.py
python scripts/run_deployment.py
uvicorn mlops_pipeline.serving:app --port 8000
```

## Tests

```bash
ruff check src tests scripts
pytest -q
```

## Failure handling

- Class imbalance or non-finite data fails `validate`.
- `roc_auc` / `recall` below gates fail `evaluate`; nothing is registered.
- Missing Production alias fails `package`.
- `/score` without a loaded model returns 503.

## Trade-offs

- Local artifact store instead of a remote object store so the graph is testable without credentials.
- Feature enrichment is a function imported by both training and serving so the contract cannot drift silently.
- Retrain is a trigger, not an automatic infinite loop.

## What I would improve next

- Remote store (S3/GCS) and locking around promotion.
- Dataset snapshots with hashes on the raw npz.
- Canary package stage that scores a holdout before flipping Production.
