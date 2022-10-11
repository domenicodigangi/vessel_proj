#%%
from typing import List, Tuple
import pandas as pd
import torch
import torch_geometric
import numpy as np
from sklearn.neighbors import BallTree
from pathlib import Path


#%%

savefold = Path("/home/digan/cnr/vessel_proj/data/processed/gnn/geolife_graphs")

savefile = savefold / "graph_list.pt"

graph_list = torch.load(savefile)

#%%
from sklearn.model_selection import train_test_split

torch.manual_seed(12345)

n_samples = len(graph_list)
train_size = round(n_samples * 0.7)
train_graphs, val_test_graphs = train_test_split(graph_list, train_size=train_size)
test_size = round(len(val_test_graphs) / 2)
val_graphs, test_graphs = train_test_split(val_test_graphs, test_size=test_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for g in train_graphs + test_graphs:
    g = g.to(device)


train_loader = torch_geometric.loader.DataLoader(train_graphs, batch_size=32)
val_loader = torch_geometric.loader.DataLoader(val_graphs, batch_size=32)
test_loader = torch_geometric.loader.DataLoader(test_graphs, batch_size=32)

# %%

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

num_classes = torch.unique(torch.tensor([g.y for g in graph_list])).shape[0]

num_node_features = torch.unique(torch.tensor([g.x.shape[1] for g in graph_list]))[
    0
].item()


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=64)
print(model)
# %%

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

model = model.to(device)


def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(
            data.x, data.edge_index, data.batch
        )  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        print(loss.data)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        print(loss.data)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# %%
