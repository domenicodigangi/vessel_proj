""" 
Start tracking data in wandb 

"""
import geopandas as gpd

from vessel_proj.data import  get_data_path, get_project_name, get_wandb_root_path

import wandb
import argh

def main():
    with wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="log local raw data", job_type ="load-data",  reinit=True) as run:

        artifact = wandb.Artifact('all_raw_data', type='dataset')
        # keep raw data locally
        artifact.add_reference('file:///' + str(get_data_path() / "raw" ))
        run.log_artifact(artifact)

        #
        artifact = wandb.Artifact('edge_list', type='dataset')
        artifact.add_reference('file:///' + str(get_data_path() / "raw" / "voyage_links.csv" ))
        run.log_artifact(artifact)

        artifact = wandb.Artifact('ports_wpi', type='dataset')
        artifact.add_reference('file:///' + str(get_data_path() / "raw" / "wpi" ))
        run.log_artifact(artifact)
        #%% load geopandas, save to csv and log ports info as artifact

        df_ports = gpd.read_file(get_data_path() / "raw" / "wpi" / 'WPI.shp' )
        ports_csv_file = get_data_path() / "raw" / 'ports_info.csv'
        df_ports.to_csv(ports_csv_file)
        artifact = wandb.Artifact('ports_info_csv', type='dataset')
        artifact.add_reference('file:///' + str(ports_csv_file))
        run.log_artifact(artifact)

argh.dispatch_command(main)

