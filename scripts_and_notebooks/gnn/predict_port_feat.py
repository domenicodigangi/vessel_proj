
# %%
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from vessel_proj.preprocess_data.gnn import get_graph_from_saved_data, get_train_mask


# %%
graph = get_graph_from_saved_data()
graph.num_classes = graph.y.unique().shape[0]
graph.x
n = 1000

graph.train_mask, graph.test_mask = get_train_mask(graph.x.shape[0])


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

# %%


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
