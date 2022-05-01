# %%
import torch_geometric as pyg
import torch
import torch_geometric
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from pathlib import Path
from vessel_proj.utils import (
    get_wandb_root_path,
    get_project_name
)

from vessel_proj.data.create_and_log_centralities import clean_group_edges
import wandb

# %% get data from artifacts
api = wandb.Api()
artifact = api.artifact(f"{get_project_name()}/edge_list:latest")
dir = artifact.checkout(root=get_wandb_root_path())
df_edges = pd.read_parquet(Path(dir) / "edge_list.parquet")
dfg_edges = clean_group_edges(df_edges)

# %%
dfg_edges

# %%


edge_index = torch.tensor(
    dfg_edges[["start_port", "end_port"]].values.astype(int)).transpose(0, 1)

g = pyg.data.Data(edge_index=edge_index, edge_attr=dfg_edges[[
                  "duration_avg_days", "trips_count", "vesseltype_count"]])
