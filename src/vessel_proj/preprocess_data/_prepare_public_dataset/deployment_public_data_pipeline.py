# %%
from prefect.deployments import DeploymentSpec

from vessel_proj.preprocess_data._prepare_public_dataset import public_data_pipeline


DeploymentSpec(
    flow=public_data_pipeline.main,
    name="public_data_pipeline",
)
