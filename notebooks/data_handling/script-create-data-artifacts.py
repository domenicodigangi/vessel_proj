""" 
Start tracking data in wandb 

"""
import geopandas as gpd

from src.data import init_wandb_run, get_data_path

import wandb

if __name__ == '__main__':
    run = init_wandb_run()

    artifact = wandb.Artifact('all_raw_data', type='dataset')
    artifact.add_reference('file:///' + str(get_data_path() / "raw" ))
    run.log_artifact(artifact)

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


