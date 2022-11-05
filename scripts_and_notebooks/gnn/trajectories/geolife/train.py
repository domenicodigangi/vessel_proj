#%%
# %load_ext dotenv
# %dotenv /home/digan/cnr/vessel_proj/vessel_proj_secrets.env

import logging
from pathlib import Path
import dotenv
import mlflow
import torch
import torch_geometric
from sklearn.model_selection import train_test_split


from vessel_proj.ds_utils import set_mlflow
from vessel_proj.gnn.train_utils import execute_one_run
from vessel_proj.gnn.models import get_gcn_model_list

dotenv.load_dotenv("/home/digan/cnr/vessel_proj/vessel_proj_secrets.env")


#%%


#%%

logger = logging.getLogger()
logger.setLevel(logging.INFO)
EXPERIMENT_VERSION = "geolife"
loadfold = Path("/home/digan/cnr/vessel_proj/data/processed/gnn/geolife_graphs")
set_mlflow()


for graph_type in [
    "spatio_temporal_percentile_1_homogeneous",
    "spatio_temporal_chain_homogeneous",
    "temporal_chain_homogeneous",
]:

    loadfile = loadfold / f"graph_list_{graph_type}.pt"

    graph_list = torch.load(loadfile)

    num_classes = torch.unique(torch.tensor([g.y for g in graph_list])).shape[0]

    num_node_features = torch.unique(torch.tensor([g.x.shape[1] for g in graph_list]))[
        0
    ].item()

    torch.manual_seed(12345)

    n_samples = len(graph_list)
    train_size = round(n_samples * 0.7)
    train_graphs, val_test_graphs = train_test_split(graph_list, train_size=train_size)
    test_size = round(len(val_test_graphs) / 2)
    val_graphs, test_graphs = train_test_split(val_test_graphs, test_size=test_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for g in train_graphs + test_graphs + val_graphs:
        if ~torch.isfinite(g.x).all():
            raise ValueError("found infinite value")
        g = g.to(device)

        if ~torch.isfinite(g.x).all():
            raise ValueError("found infinite value in gPU version")

    train_loader = torch_geometric.loader.DataLoader(train_graphs, batch_size=32)
    val_loader = torch_geometric.loader.DataLoader(val_graphs, batch_size=32)
    test_loader = torch_geometric.loader.DataLoader(test_graphs, batch_size=32)

    experiment_name = f"classify_trajectories_{EXPERIMENT_VERSION}"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_list = get_gcn_model_list(num_node_features, num_classes)
    lr_values = [0.005]
    n_epochs = 250
    for model in model_list:
        execute_one_run(
            experiment, model, train_loader, val_loader, lr_values, graph_type, n_epochs
        )
