#%%
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from vessel_proj.preprocess_data.traj_to_graphs import get_chain_graph_from_traj

loadfoldpath = Path(
    "/home/digan/cnr/vessel_proj/data/interim/vessel_test_traj_preprocessed"
)
savefold = Path("/home/digan/cnr/vessel_proj/data/processed/gnn/vessel_test_graphs")
savefold.mkdir(exist_ok=True)

len(list(loadfoldpath.iterdir()))

graph_list = []
for filepath in loadfoldpath.iterdir():

    df_traj = pd.read_parquet(filepath)

    graph = get_chain_graph_from_traj(
        df_traj.iloc[1:, :], cat_col="cat", additional_columns=["sog", "cog", "heading"]
    )
    graph_list.append(graph)

df_traj["cat"]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
all_classes = np.unique([g.y for g in graph_list])
le.fit(all_classes)
for graph in graph_list:

    graph.y = torch.tensor(le.transform([graph.y]))


savefile = savefold / "graph_list.pt"

torch.save(graph_list, savefile)

# %%
