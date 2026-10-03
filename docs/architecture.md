# Architecture

Training and deployment share an `ArtifactStore` and disagree on everything else.

- Training writes `var/store/runs/<id>/` and a new registry version.
- Promotion updates aliases in `var/store/registry/index.json`.
- Deployment copies the Production artifact into `var/store/serving-bundle/`.
- The API loads the Production alias (or the bundle) and applies the same `enrich()` used at train time.

The Airflow DAG is a thin wrapper so a platform team can schedule the same callables.

## Rollback

Point `PIPELINE_SERVING_STAGE` at a previous alias or re-promote an older version, then restart the API process.
