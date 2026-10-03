from mlops_pipeline.pipelines import run_deployment

if __name__ == "__main__":
    ctx = run_deployment()
    print({"bundle": ctx.values.get("bundle_dir"), "version": ctx.values.get("served_version")})
