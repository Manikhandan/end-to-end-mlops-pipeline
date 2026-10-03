from __future__ import annotations

from mlops_pipeline.exceptions import TriggerError
from mlops_pipeline.runtime import PipelineContext


def monitor(ctx: PipelineContext) -> dict:
    """Reads a drift payload produced by a monitoring job. Does not invent live traffic."""
    psi = float(ctx.values.get("observed_psi", 0.0))
    error_rate = float(ctx.values.get("observed_error_rate", 0.0))
    return {"observed_psi": psi, "observed_error_rate": error_rate}


def retrain_trigger(ctx: PipelineContext) -> dict:
    psi = float(ctx.require("observed_psi"))
    threshold = float(ctx.require("drift_psi_trigger"))
    if psi >= threshold:
        return {"retrain": True, "reason": f"psi {psi} >= {threshold}"}
    if float(ctx.require("observed_error_rate")) >= 0.15:
        return {"retrain": True, "reason": "error_rate"}
    raise TriggerError("no retrain condition met")
