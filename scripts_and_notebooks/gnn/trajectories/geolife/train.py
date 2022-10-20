#%%
# %load_ext dotenv
# %dotenv /home/digan/cnr/vessel_proj/vessel_proj_secrets.env

import logging
import tempfile
from pathlib import Path
import dotenv
import mlflow
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv, global_mean_pool
from vessel_proj.ds_utils import set_mlflow
from vessel_proj.ds_utils.torch.opt import grad_norm_from_list

dotenv.load_dotenv("/home/digan/cnr/vessel_proj/vessel_proj_secrets.env")

#%%

logger = logging.getLogger()
logger.setLevel(logging.INFO)
EXPERIMENT_VERSION = "test_geolife"
loadfold = Path("/home/digan/cnr/vessel_proj/data/processed/gnn/geolife_graphs")

loadfile = loadfold / "graph_list.pt"

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

# %%


class GCN5(torch.nn.Module):
    def __init__(self, hidden_channels):
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
    def __init__(self, hidden_channels):
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
    def __init__(self, hidden_channels):
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
    def __init__(self, hidden_channels):
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


def get_accuracy(model: torch.nn.Module, data: torch_geometric.data.Data) -> float:
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    model.train()
    return acc


def train(model, train_loader, optimizer):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.
        out = model(
            data.x, data.edge_index, data.batch
        )  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        # print(loss.data)

    return loss


def get_accuracy(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


# %%


set_mlflow()

experiment_name = f"classify_trajectories_{EXPERIMENT_VERSION}"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_list = [
    GCN2(hidden_channels=64),
    GCN2(hidden_channels=32),
    GCN3(hidden_channels=64),
    GCN3(hidden_channels=32),
    GCN4(hidden_channels=64),
    GCN4(hidden_channels=32),
    GCN5(hidden_channels=64),
    GCN5(hidden_channels=32),
]

criterion = torch.nn.CrossEntropyLoss()


def execute_one_run(model):

    for lr in [0.0005]:

        with mlflow.start_run(experiment_id=experiment.experiment_id):
            model = model.to(device)

            h_par = {"optim": "Adam"}
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            h_par = {
                k: v for k, v in optimizer.param_groups[0].items() if k not in "params"
            }
            h_par["optim_str"] = optimizer.__str__()
            h_par["n_epochs"] = 500
            h_par["Model Name"] = model._get_name()
            h_par["hidden_channels"] = model.hidden_channels
            h_par["Model String"] = model.__str__()

            mlflow.log_params(h_par)

            with tempfile.TemporaryDirectory() as tmpdirname:

                tmp_path = Path(tmpdirname)
                tb_fold = tmp_path / "tb_logs"
                tb_fold.mkdir(exist_ok=True)
                logger.info(f"tensorboard logs in {tb_fold}")
                checkpoints_fold = tmp_path / "checkpoints"
                checkpoints_fold.mkdir(exist_ok=True)
                writer = SummaryWriter(str(tb_fold))

                # log all files and sub-folders in temp fold as artifacts
                for epoch in range(h_par["n_epochs"]):

                    loss = train(model, train_loader, optimizer)

                    train_accuracy = get_accuracy(train_loader)
                    valid_accuracy = get_accuracy(val_loader)
                    print(
                        f"Epoch: {epoch:03d}, Train Acc: {train_accuracy:.4f}, Validation Acc: {valid_accuracy:.4f}"
                    )

                    writer.add_scalar("Loss/value", loss.item(), epoch)
                    writer.add_scalar("Valid Accuracy", valid_accuracy, epoch)
                    writer.add_scalar("Train Accuracy", train_accuracy, epoch)

                    grad_norm = grad_norm_from_list(model.parameters())
                    writer.add_scalar("Loss/grad_norm", grad_norm, epoch)

                    if epoch % 1000 == 0:
                        logger.info(f"Saving checkpoint {epoch}")
                        filepath_checkpoint = (
                            checkpoints_fold / f"checkpoint_{epoch}.pt"
                        )
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss,
                                "valid_accuracy": valid_accuracy,
                                "train_accuracy": train_accuracy,
                            },
                            filepath_checkpoint,
                        )

                        mlflow.log_artifact(filepath_checkpoint)

                mlflow.log_metrics(
                    {"Loss/value": loss.item(), "Valid Accuracy": valid_accuracy}
                )

                mlflow.log_artifacts(tmp_path)

    mlflow.end_run()


for model in model_list:
    execute_one_run(model)

# from joblib import Parallel, delayed

# results = Parallel(n_jobs=2)(delayed(execute_one_run)(model) for model in model_list)
