from typing import List
from itertools import product

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    HeteroConv,
    SAGEConv,
    global_mean_pool,
    HANConv,
)


class GCN5(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_node_features: int, num_classes: int):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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


class GCNHomogeneous(torch.nn.Module):
    def __init__(self, num_layers: int, hidden_channels: int, num_classes: int):
        torch.manual_seed(12345)
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.lin = Linear(hidden_channels, num_classes)
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GCNConv(-1, hidden_channels)
            self.convs.append(conv)

    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()

        x = self.convs[-1](x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCNHeterogeneous(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_channels: int,
        num_classes: int,
        rel_name_list: List[str],
        aggr_het: str = "sum",
    ):

        super().__init__()
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("traj_point", rel_name, "traj_point"): GCNConv(-1, hidden_channels)
                    for rel_name in rel_name_list
                },
                aggr=aggr_het,
            )
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x_dict, edge_index_dict, batch):
        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        x_dict = self.convs[-1](x_dict, edge_index_dict)

        # 2. Readout layer
        for k in x_dict.keys():
            x_dict[k] = global_mean_pool(
                x_dict[k], batch[k]
            )  # [batch_size, hidden_channels]

            x_dict[k] = F.dropout(x_dict[k], p=0.5, training=self.training)

            x_dict[k] = self.lin(x_dict[k])

        return x_dict


class HANHeterogeneous(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_channels: int,
        num_classes: int,
        rel_name_list: List[str],
    ):

        super().__init__()
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        metadata = (
            ["traj_point"],
            [("traj_point", rel_name, "traj_point") for rel_name in rel_name_list],
        )
        for _ in range(num_layers):
            conv = HANConv(-1, hidden_channels, metadata)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x_dict, edge_index_dict, batch):
        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        x_dict = self.convs[-1](x_dict, edge_index_dict)

        # 2. Readout layer
        for k in x_dict.keys():
            x_dict[k] = global_mean_pool(
                x_dict[k], batch[k]
            )  # [batch_size, hidden_channels]

            x_dict[k] = F.dropout(x_dict[k], p=0.5, training=self.training)

            x_dict[k] = self.lin(x_dict[k])

        return x_dict


def get_gcn_homogeneous_model_list(
    n_layers_list: List[int],
    hidden_channels_list: List[int],
    num_classes: int,
) -> List:

    gcn_model_list = [
        GCNHomogeneous(n_layers, hidden_channels, num_classes)
        for n_layers, hidden_channels in product(n_layers_list, hidden_channels_list)
    ]

    # gcn_model_list = [
    #     # GCNHomogeneous(2, 32, num_classes),
    #     GCN2(32, num_node_features, num_classes),
    #     GCN2(64, num_node_features, num_classes),
    #     GCN3(64, num_node_features, num_classes),
    #     GCN3(32, num_node_features, num_classes),
    #     GCN4(64, num_node_features, num_classes),
    #     GCN4(32, num_node_features, num_classes),
    #     GCN5(64, num_node_features, num_classes),
    #     GCN5(32, num_node_features, num_classes),
    # ]
    return gcn_model_list


def get_gcn_heterogeneous_model_list(
    n_layers_list: List[int],
    hidden_channels_list: List[int],
    num_classes: int,
    rel_name_list: List[str],
    aggr_het: str = "sum",
) -> List:

    gcn_model_list = [
        GCNHeterogeneous(
            n_layers, hidden_channels, num_classes, rel_name_list, aggr_het=aggr_het
        )
        for n_layers, hidden_channels in product(n_layers_list, hidden_channels_list)
    ]

    han_model_list = [
        HANHeterogeneous(n_layers, hidden_channels, num_classes, rel_name_list)
        for n_layers, hidden_channels in product(n_layers_list, hidden_channels_list)
    ]

    return han_model_list  # + gcn_model_list
