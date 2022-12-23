# ImPortance
==============================

A project to learn explainability methods based on shapley values by applying them to a classification task.

## Data and artifacts' management
We use wandb to handle artifacts and log models' performances etc..

- When we want to use wandb we need to init a run (auto linked to the vessel proj) with wandb.init(project=get_project_name(), name=name, dir=get_wandb_root_path(), job_type=job_type, group=group, reinit=True)

    logger.info(f"Logging to wandb run called {run.name}")

- The initial data artifacts are created in notebooks/data_handling/1.0-ddg-create-data-artifacts-py . 
- Artifacts are basically versioned folders, that can be stored locally or in the free wandb cloud. Creating artifacts is useful to track data versions. 
### using artifacts
- Within a run,  single files can be easily loaded by 'src.data.get_file_from_artifact("edge_list:latest", run=run)'
- Outside runs use 'src.data.get_file_from_artifact("edge_list:latest")'


