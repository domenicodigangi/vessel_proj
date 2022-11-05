#%%
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from vessel_proj.preprocess_data.traj_to_graphs import (
    get_chain_graph_from_traj,
    get_homogeneous_spatio_temp_chain_graph,
    get_homogeneous_spatio_temp_closest_percentile_graph,
    get_het_graph_from_df_traj_and_df_feat,
)

#%%
loadfoldpath = Path(
    "/home/digan/cnr/vessel_proj/data/interim/geolife_traj_preprocessed"
)


graph_list = []
for filepath in loadfoldpath.iterdir():

    df_traj = pd.read_parquet(filepath)
    additional_columns = [
        "altitude",
    ]

    graph = get_het_graph_from_df_traj_and_df_feat(
        df_traj, cat_col="transportation_mode", additional_columns=additional_columns
    )

    graph_list.append(graph)


le = LabelEncoder()
all_classes = np.unique([g.y for g in graph_list])
le.fit(all_classes)
import copy

for graph in graph_list:

    y = torch.tensor(le.transform([graph.y]))

    graph.y = y


savefold = Path("/home/digan/cnr/vessel_proj/data/processed/gnn/geolife_graphs")

graph_type = "spatio_temporal_chain_heterogeneous"
savefile = savefold / f"graph_list_{graph_type}.pt"

torch.save(graph_list, savefile)

# %%
percentile = 1
graph_list = []
for filepath in loadfoldpath.iterdir():

    df_traj = pd.read_parquet(filepath)
    additional_columns = [
        "altitude",
    ]

    graph = get_homogeneous_spatio_temp_closest_percentile_graph(
        df_traj,
        cat_col="transportation_mode",
        additional_columns=additional_columns,
        percentile=percentile,
    )

    graph_list.append(graph)


le = LabelEncoder()
all_classes = np.unique([g.y for g in graph_list])
le.fit(all_classes)
import copy

for graph in graph_list:

    y = torch.tensor(le.transform([graph.y]))

    graph.y = y


savefold = Path("/home/digan/cnr/vessel_proj/data/processed/gnn/geolife_graphs")

graph_type = f"spatio_temporal_percentile_{percentile}_heterogeneous"
savefile = savefold / f"graph_list_{graph_type}.pt"

torch.save(graph_list, savefile)

# %%
