import continual
import os

client = continual.Client()

run_id = os.environ.get("CONTINUAL_RUN_ID", None)
run = client.runs.create(description="Create and promote model", run_id=run_id)

model = run.models.create("test_model")
model_version = model.model_versions.create()

accuracy_metric = model_version.metrics.create(id="accuracy", direction="HIGHER")
accuracy_metric.log(value=0.8)

model.promotions.create(
	model_version_name=model_version.name, reason="UPLIFT"
)

run.complete()
