
# %%
import numpy as np
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx

# %% [markdown]
# ## Data Handling in PyG

# %% [markdown]
# ### Data

# %% [markdown]
# Let's create a dummy graph

# %%
embeddings = torch.rand((100, 16), dtype=torch.float)

# %%
rows = np.random.choice(100, 500)
cols = np.random.choice(100, 500)
edges = torch.tensor([rows, cols])
edges.shape
# %%
edges_attr = np.random.choice(3,500)

# %%
ys = torch.rand((100)).round().long()

# %% [markdown]
# Convert the graph information into a PyG Data object

# %%
graph = data.Data(x=embeddings, edge_index=edges, edge_attr=edges_attr, y=ys)

# %%
graph

# %% [markdown]
# Let's visualize the information contained in the data object

# %%
for prop in graph:
    print(prop)

# %%
vis = to_networkx(graph)

node_labels = graph.y.numpy()

import matplotlib.pyplot as plt
plt.figure(1,figsize=(15,13)) 
nx.draw(vis, cmap=plt.get_cmap('Set3'),node_color = node_labels,node_size=70,linewidths=6)
plt.show()

# %% [markdown]
# ### Batch

# %% [markdown]
# With the Batch object we can represent multiple graphs as a single disconnected graph

# %%
graph2 = graph

# %%
batch = data.Batch().from_data_list([graph, graph2])

# %%
print("Number of graphs:",batch.num_graphs)
print("Graph at index 1:",batch[1])
print("Retrieve the list of graphs:\n",len(batch.to_data_list()))

# %% [markdown]
# ### Cluster

# %% [markdown]
# ClusterData groups the nodes of a graph into a specific number of cluster for faster computation in large graphs, then use ClusterLoader to load batches of clusters

# %%
#cluster = data.ClusterData(graph, 5)

# %%
#clusterloader = data.ClusterLoader(cluster)

# %% [markdown]
# ### Sampler

# %% [markdown]
# For each convolutional layer, sample a maximum of nodes from each neighborhood (as in GraphSAGE)

# %%
sampler = data.NeighborSampler(graph.edge_index, sizes=[3,10], batch_size=4,
                                  shuffle=False)

# %%
for s in sampler:
    print(s)
    break

# %%
print("Batch size:", s[0])
print("Number of unique nodes involved in the sampling:",len(s[1]))
print("Number of neighbors sampled:", len(s[2][0].edge_index[0]), len(s[2][1].edge_index[0]))

# %% [markdown]
# ### Datasets

# %% [markdown]
# List all the available datasets

# %%
datasets.__all__

# %%
name = 'Cora'
transform = transforms.Compose([
    transforms.AddTrainValTestMask('train_rest', num_val=500, num_test=500),
    transforms.TargetIndegree(),
])
cora = datasets.Planetoid('./data', name, pre_transform=transforms.NormalizeFeatures(), transform=transform)

# %%
aids = datasets.TUDataset(root="./data", name="AIDS")

# %%
print("AIDS info:")
print('# of graphs:', len(aids))
print('# Classes (graphs)', aids.num_classes)
print('# Edge features', aids.num_edge_features)
print('# Edge labels', aids.num_edge_labels)
print('# Node features', aids.num_node_features)

# %%
print("Cora info:")
print('# of graphs:', len(cora))
print('# Classes (nodes)', cora.num_classes)
print('# Edge features', cora.num_edge_features)
print('# Node features', cora.num_node_features)

# %%
aids.data

# %%
aids[0]

# %%
cora.data

# %%
cora[0]

# %%
cora_loader = data.DataLoader(cora)

# %%
for l in cora_loader:
    print(l)
    break

# %%



