# %%
%load_ext dotenv
%dotenv /home/digan/cnr/vessel_proj/vessel_proj_secrets.env

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv
import torch_geometric
import torch.nn.functional as F
import torch
from vessel_proj.preprocess_data.gnn import get_graph_from_saved_data, get_train_mask
from vessel_proj.ds_utils.torch.opt import grad_norm_from_list
import mlflow
from vessel_proj.ds_utils import set_mlflow
from torch.utils.tensorboard import SummaryWriter
import tempfile
from pathlib import Path
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
EXPERIMENT_VERSION = "test"
# %%


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

class SAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(graph.num_node_features, 16)
        self.conv2 = SAGEConv(16, graph.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(graph.num_node_features, 16)
        self.conv2 = GATConv(16, graph.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GATv2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(graph.num_node_features, 16)
        self.conv2 = GATv2Conv(16, graph.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def get_accuracy(model: torch.nn.Module, data: torch_geometric.data.Data) ->float:
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    model.train()
    return acc


# %%


set_mlflow()

experiment_name = f"predict_node_feat_{EXPERIMENT_VERSION}"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graph = get_graph_from_saved_data()
graph.num_classes = graph.y.unique().shape[0]
graph.train_mask, graph.val_mask, graph.test_mask = get_train_mask(graph.x.shape[0])

mlflow.end_run()
for model in [GCN(), SAGE(), GAT(), GATv2()]:
    with mlflow.start_run(experiment_id=experiment.experiment_id): 
        model = model.to(device)
        data = graph.to(device)

        h_par = {"optim": "Adam"}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        h_par = {k: v for k, v in optimizer.param_groups[0].items() if k not in "params"}
        h_par["optim_str"] = optimizer.__str__()
        h_par["n_epochs"] = 1000
        h_par["Model Name"] = model._get_name()
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

            model.train()
            for epoch in range(h_par["n_epochs"]):
                optimizer.zero_grad()
                out = model(data)
                loss = F.nll_loss(out[data.train_mask, :], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                valid_accuracy = get_accuracy(model, data)


                writer.add_scalar("Loss/value", loss.item(), epoch)
                writer.add_scalar("Valid Accuracy", valid_accuracy, epoch)

                grad_norm = grad_norm_from_list(model.parameters())
                writer.add_scalar("Loss/grad_norm", grad_norm, epoch)


                if epoch % 1000 == 0:
                    logger.info(f"Saving checkpoint {epoch}")
                    filepath_checkpoint = checkpoints_fold / f"checkpoint_{epoch}.pt"
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss,
                                'valid_accuracy': valid_accuracy, 
                                }, filepath_checkpoint) 
            
                    mlflow.log_artifact(filepath_checkpoint)

            mlflow.log_metrics({"Loss/value": loss.item(), "Valid Accuracy": valid_accuracy})

            mlflow.log_artifacts(tmp_path)


# %%
