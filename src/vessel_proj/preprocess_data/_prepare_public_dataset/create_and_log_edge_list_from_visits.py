
#%%
from vessel_proj.preprocess_data import  get_data_path, get_project_name, get_project_root, get_wandb_root_path, create_data_fold_structure, shaper_slow
import wandb
import pandas as pd
from pathlib import Path
import argh
from argh import arg
import logging
from prefect import task

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

#%%


if False:
    run = wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="log local raw data", job_type ="test",  reinit=True)
    nrows = 1000

@task
@arg("--nrows", help="limit to the first nrows rows", type=int)
def main(nrows=None):
    with wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="log local raw data", job_type ="load-data",  reinit=True) as run:

        api = wandb.Api()
        proj_name = get_project_name()
        dir = api.artifact(f"{proj_name}/visits-augmented-cleaned:latest").checkout(root=get_wandb_root_path()/"visits-augmented-cleaned")


        input_file = Path(dir) / 'visits-augmented-cleaned.parquet'

        output_file = get_data_path() / "interim" /'edge_list.parquet'

        df_visits = pd.read_parquet(input_file)

        if nrows is not None:
            df_visits = df_visits[:nrows]            
            output_file = Path(output_file)
            no_suff_name = output_file.name.replace(output_file.suffix, '')
            output_file = Path(output_file.parent) / ( f"{no_suff_name}_from_{nrows}_visits{output_file.suffix}"  )


        df_edges = shaper_slow(df_visits, output_file=output_file)

        wandb.log_artifact(str(output_file), name='edge_list', type='dataset') 



argh.dispatch_command(main)

