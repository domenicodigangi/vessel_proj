#%%
import torch_geometric
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from pathlib import Path
from vessel_proj.data import (
    get_one_file_from_artifact,
    get_wandb_root_path, 
    get_project_name
)

from vessel_proj.data.create_and_log_centralities import clean_group_edges
import wandb

#%% get data from artifacts
api = wandb.Api()
artifact = api.artifact(f"{get_project_name()}/edge_list:latest")
dir = artifact.checkout(root=get_wandb_root_path())
df_edges = pd.read_parquet(Path(dir) / "edge_list.parquet")
dfg_edges = clean_group_edges(df_edges)

#%%
dfg_edges

#%%
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
