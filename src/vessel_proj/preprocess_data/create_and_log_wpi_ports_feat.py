
#%%
import pandas as pd
from matplotlib import pyplot as plt
import wandb
from pathlib import Path
from vessel_proj.preprocess_data import  get_wandb_root_path, get_project_name, get_data_path
import geopandas as gpd
import argh
from prefect import task


# %%

if False:
    run = wandb.init(project=get_project_name(), name="create_port_feat_wpi", dir=get_wandb_root_path(), group="data_preprocessing", reinit=True)
@task
def main():
    """load world port index info, cast types, drop some columns and store as parquet"""
    with wandb.init(project=get_project_name(), name="create_port_feat_wpi", dir=get_wandb_root_path(), group="data_preprocessing", reinit=True) as run:


        proj_name = get_project_name()
        dir = run.use_artifact(f"{proj_name}/all_raw_data:latest").checkout(root=get_wandb_root_path()/"all_raw_data")

        df_ports = gpd.read_file(dir / 'WPI.shp' )
        
        df_ports_clean = clean_ports_info(df_ports)
        art_ports_clean = wandb.Artifact("ports_features", type="dataset", description="df with different port features from world port index")
        with art_ports_clean.new_file('ports_features.parquet', mode='wb') as file:
            df_ports_clean.to_parquet(file)
        
        run.log_artifact(art_ports_clean)



def clean_ports_info(df_ports):

    df_ports.describe().transpose()
    df_ports.describe(include = ["object"]).transpose()

    col_to_drop = ["CHART", "geometry", "LAT_DEG", "LAT_MIN", "LONG_DEG", "LONG_MIN", "LAT_HEMI", "LONG_HEMI"]

    col_single_val = df_ports.columns[df_ports.apply(lambda x: pd.unique(x).shape[0]) == 1].values.tolist()
    print(col_single_val)
    
    col_to_drop.extend(col_single_val)

    df_ports.drop(columns=col_to_drop, inplace=True)
    
    
    for c in df_ports.select_dtypes(include=['object']).columns:
        df_ports[c] = df_ports[c].astype('category')
    

    df_ports["INDEX_NO"] = df_ports["INDEX_NO"].astype('int')
    df_ports = df_ports.set_index("INDEX_NO")

    # %% EDA ports info
    if False:
        for col in df_ports.columns:
            if col not in ["PORT_NAME", "COUNTRY"]:
                plt.figure()
                df_ports[col].hist()
                plt.title(col)
                plt.show()

    return df_ports

argh.dispatch_command(main)

