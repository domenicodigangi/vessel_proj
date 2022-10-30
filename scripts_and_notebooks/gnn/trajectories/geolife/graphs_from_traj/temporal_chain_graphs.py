#%%
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from vessel_proj.preprocess_data.traj_to_graphs import get_chain_graph_from_traj

loadfoldpath = Path(
    "/home/digan/cnr/vessel_proj/data/interim/geolife_traj_preprocessed"
)

len(list(loadfoldpath.iterdir()))

graph_list = []
for filepath in loadfoldpath.iterdir():

    df_traj = pd.read_parquet(filepath)
    additional_columns = [
        "altitude",
    ]

    graph = get_chain_graph_from_traj(
        df_traj, cat_col="transportation_mode", additional_columns=additional_columns
    )

    graph_list.append(graph)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
all_classes = np.unique([g.y for g in graph_list])
le.fit(all_classes)
for graph in graph_list:

    graph.y = torch.tensor(le.transform([graph.y]))


savefold = Path("/home/digan/cnr/vessel_proj/data/processed/gnn/geolife_graphs")

savefile = savefold / "graph_list_temporal_chain_homogeneous.pt"

torch.save(graph_list, savefile)

# %%
