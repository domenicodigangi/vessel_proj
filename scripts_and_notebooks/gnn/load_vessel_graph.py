
# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.datasets as datasets
import torch_geometric
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import pandas as pd
from vessel_proj.preprocess_data.get_and_save_centralities_from_graph_data import get_graph_data

from vessel_proj.task.classification_task_pipeline_ports_centr import encode_features, simple_impute_cols
from vessel_proj.preprocess_data import get_latest_port_data_task
# %% get links data
vessel_category = "cargo"
idx = pd.IndexSlice
trips_all = get_graph_data()
trips_cat = trips_all.loc[idx[:, :, vessel_category]]
#%% get features
data_ports = get_latest_port_data_task.fn(vessel_category)
feat_names_non_cat=["TIDE_RANGE", "LATITUDE", "LONGITUDE"]
feat_enc = encode_features.fn(data_ports, feat_names_non_cat=feat_names_non_cat, cols_to_drop=["PORT_NAME", "REGION_NO", "PUB"],)["features"]

feat_imputed, _ = simple_impute_cols(feat_names_non_cat, feat_enc) 

#%% clean indices as integers from 1 to N using the ordering of the features df

clean_inds = (
    feat_imputed
    .reset_index()["INDEX_NO"]
    .reset_index()
    .rename(columns={"index": "new_ind", "INDEX_NO": "old_ind"})
)

trips_new_inds = (
    trips_cat.reset_index()
    .merge(clean_inds, how="inner", left_on="start_port", right_on="old_ind", validate="many_to_one")
    .rename(columns={"new_ind": "start_port_new_ind"})
    .merge(clean_inds, how="inner", left_on="end_port", right_on="old_ind", validate="many_to_one")
    .rename(columns={"new_ind": "end_port_new_ind"})
    .drop(columns=["old_ind_x", "old_ind_y", "start_port", "end_port"])
)

# %%
target_feat = "COUNTRY"
edges = torch.tensor(trips_new_inds[["start_port_new_ind", "end_port_new_ind"]].to_numpy().T, dtype=torch.long)
edges.shape

y = torch.tensor(feat_imputed[target_feat].to_numpy()).long()
feat = torch.tensor(feat_imputed.drop(columns=target_feat).to_numpy()).float()
feat.shape

edges_attr = torch.tensor(trips_new_inds[["trips_count"]].to_numpy().T).float()
# %%  Convert the graph information into a PyG Data object
graph = torch_geometric.data.Data(x=feat, edge_index=edges, edge_attr=edges_attr, y=y)
graph.y
# %% 

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

import torch.nn.functional as F
from torch_geometric.nn import GCNConv



#%%
graph.num_classes = graph.y.unique().shape[0]
graph.x
n=1000
def get_train_mask(n, test_fract=0.2):
    torch.manual_seed(0)
    bern = torch.distributions.bernoulli.Bernoulli(torch.tensor(test_fract))
    train_mask = bern.sample((n,)).bool()
    test_mask = torch.logical_not(train_mask)
    return train_mask, test_mask

graph.train_mask, graph.test_mask = get_train_mask(graph.x.shape[0])
graph.train_mask.shape
#%%

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(graph.num_node_features, 16)
        self.conv2 = GCNConv(16, graph.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def get_accuracy(model, data):
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    model.train()
    return acc

#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = graph.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask, :], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    acc = get_accuracy(model, data)    
    print(f'Accuracy: {acc:.4f}')
# %%
# %%
F.log_softmax(model(data), dim=1).shape

model(data).shape

data.y
#
