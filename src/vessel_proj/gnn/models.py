from typing import List

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv, global_mean_pool


class GCN5(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_node_features: int, num_classes: int):
        super(GCN5, self).__init__()
        torch.manual_seed(12345)
        self.hidden_channels = hidden_channels
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.conv5(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCN4(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_node_features: int, num_classes: int):
        super(GCN4, self).__init__()
        torch.manual_seed(12345)
        self.hidden_channels = hidden_channels
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCN3(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_node_features: int, num_classes: int):
        super(GCN3, self).__init__()
        torch.manual_seed(12345)
        self.hidden_channels = hidden_channels
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


class GCN2(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_node_features: int, num_classes: int):
        super(GCN2, self).__init__()
        torch.manual_seed(12345)
        self.hidden_channels = hidden_channels
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


def get_gcn_model_list(num_node_features: int, num_classes: int) -> List:
    gcn_model_list = [
        GCN2(64, num_node_features, num_classes),
        GCN2(32, num_node_features, num_classes),
        GCN3(64, num_node_features, num_classes),
        GCN3(32, num_node_features, num_classes),
        GCN4(64, num_node_features, num_classes),
        GCN4(32, num_node_features, num_classes),
        GCN5(64, num_node_features, num_classes),
        GCN5(32, num_node_features, num_classes),
    ]
    return gcn_model_list
