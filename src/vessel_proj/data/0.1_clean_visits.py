""" 
Start tracking data in wandb 

"""
#%%
from vessel_proj.data import  get_data_path, get_project_name, get_project_root, get_wandb_root_path, create_data_fold_structure
import wandb
import pandas as pd
from pathlib import Path
import argh
from argh import arg
import logging
logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
#%%
if False:
    run = wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="clean_data", job_type ="clean-data",  reinit=True)
    min_dur_sec=300
    types_to_drop = ["fishing", "tug tow"]

@arg("--min_dur_sec", help="Drop all visits that lasted less than")
def main(min_dur_sec=300, types_to_drop = ["fishing", "tug tow"]):
    with wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="clean_data", job_type ="clean-data",  reinit=True) as run:

        api = wandb.Api()
        proj_name = get_project_name()
        dir = api.artifact(f"{proj_name}/all_raw_data:latest").checkout(root=get_wandb_root_path()/"all_raw_data")

        df = pd.read_parquet(Path(dir) / 'visits-augmentend.parquet')

        run.log({"min_duration_seconds_allowed": min_dur_sec})

        inds = df["duration_seconds"] < min_dur_sec
        logger.info(f"Dropping {inds.sum()} rows for too low duration")
        df.drop(df[inds].index, inplace=True)

        
        inds = df["vessel_category"].apply(lambda x: x in types_to_drop)
        logger.info(f"Dropping {inds.sum()} rows because vessel type in {types_to_drop}")
        df.drop(df[inds].index, inplace=True)


        # store 
        file_path = get_data_path() / "interim" / 'visits-augmented-cleaned.parquet'
        df.to_parquet(file_path)
        wandb.log_artifact(str(file_path), name='visits-augmented-cleaned', type='dataset') 



parser = argh.ArghParser()
parser.set_default_command(main)

if __name__ == "__main__":
    parser.dispatch()


