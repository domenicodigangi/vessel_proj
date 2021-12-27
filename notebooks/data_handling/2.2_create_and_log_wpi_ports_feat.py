
#%%
from networkx.classes.function import density
from networkx.classes.graphviews import generic_graph_view
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import wandb
from pathlib import Path
from vessel_proj.data import read_edge_list, get_wandb_root_path, get_project_name

import argh

# %%

def clean_ports_info(df_ports):

    descr_num = df_ports.describe().transpose()
    descr_obj = df_ports.describe(include = ["object"]).transpose()


    col_to_drop = ["CHART", "geometry", "LAT_DEG", "LAT_MIN", "LONG_DEG", "LONG_MIN", "LAT_HEMI", "LONG_HEMI"]

    col_single_val = df_ports.columns[df_ports.apply(lambda x: pd.unique(x).shape[0]) == 1].values.tolist()
    print(col_single_val)

    col_to_drop.extend(col_single_val)

    df_ports = df_ports.drop(columns=col_to_drop)

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

def wpi_features():
    """load world port index info, cast types, drop some columns and store as parquet"""
    with wandb.init(project=get_project_name(), name="create_port_feat_wpi", dir=get_wandb_root_path(), group="data_preprocessing", reinit=True) as run:

        art_ports = run.use_artifact(f"{get_project_name()}/ports_info:latest")
        dir = art_ports.download(root=get_wandb_root_path())
        df_ports = pd.read_csv(Path(dir) / "ports_info.csv")

        df_ports_clean = clean_ports_info(df_ports)
        art_ports_clean = wandb.Artifact("ports_features", type="dataset", description="df with different port features from world port index")
        with art_ports_clean.new_file('ports_features.parquet', mode='wb') as file:
            df_ports_clean.to_parquet(file)
        
        run.log_artifact(art_ports_clean)


argh.dispatch_command(wpi_features)

