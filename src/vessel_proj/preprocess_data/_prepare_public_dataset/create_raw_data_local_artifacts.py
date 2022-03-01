""" 
Start tracking data in wandb 

"""
#%%
from vessel_proj.preprocess_data import  get_data_path, get_project_name, get_project_root, get_wandb_root_path, create_data_fold_structure
import wandb
import pandas as pd
import argh
from prefect import task

#%%
if False:
    create_data_fold_structure(get_project_root())
    run = wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="log local raw data", job_type ="load-data",  reinit=True)
@task
def main():
    with wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="log local raw data", job_type ="load-data",  reinit=True) as run:

        artifact = wandb.Artifact('all_raw_data', type='dataset')

        # store csv as parquet
        dir = get_data_path() / "raw" 
        df_visits = pd.read_csv(dir / 'visits-augmented.csv')
        df_visits.to_parquet(dir / 'visits-augmented.parquet')

        artifact.add_reference('file://' + str(get_data_path() / "raw" ))

        run.log_artifact(artifact)

argh.dispatch_command(main)

