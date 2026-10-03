from mlops_pipeline.pipelines import run_training

if __name__ == "__main__":
    ctx = run_training()
    print({"run_id": ctx.run_id, "version": ctx.values.get("version"), "metrics": ctx.values.get("metrics")})
