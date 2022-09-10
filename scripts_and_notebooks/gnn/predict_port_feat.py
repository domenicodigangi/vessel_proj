# %%
%load_ext dotenv
%dotenv /home/digan/cnr/vessel_proj/vessel_proj_secrets.env

from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from vessel_proj.preprocess_data.gnn import get_graph_from_saved_data, get_train_mask
import mlflow
from vessel_proj.ds_utils import set_mlflow
from torch.utils.tensorboard import SummaryWriter


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


def get_accuracy(model, data):
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    model.train()
    return acc


# %%


set_mlflow()

experiment_name = f"predict_node_feat_{EXPERIMENT_VERSION}"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

with mlflow.start_run(experiment_id=experiment.experiment_id) as run:

    graph = get_graph_from_saved_data()
    graph.num_classes = graph.y.unique().shape[0]
    graph.x
    n_train = 1000

    graph.train_mask, graph.test_mask = get_train_mask(graph.x.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN().to(device)
    data = graph.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


    with tempfile.TemporaryDirectory() as tmpdirname:

        tmp_path = Path(tmpdirname)
        tb_fold = tmp_path / "tb_logs"
        tb_fold.mkdir(exist_ok=True)
        tb_log_str = f"saving to tb logs to {tb_fold}"
        writer = SummaryWriter(str(tb_fold))

        mlflow.log_params(parent_runs_par)
        mlflow.log_metrics()

        torch.save(run_data_dict["Y_reference"], dgp_fold / "Y_reference.pt")
                            
        # log all files and sub-folders in temp fold as artifacts
        mlflow.log_artifacts(tmp_path)

        model.train()
        for epoch in range(n_train):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask, :], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            valid_accuracy = get_accuracy(model, data)


            writer.add_scalar("Loss/value", loss.item(), epoch)
            writer.add_scalar("Valid Accuracy", valid_accuracy, epoch)
            writer.add_scalar("Loss/grad_norm", grad_norm, epoch)
            writer.add_scalar("Loss/grad_max", grad_norm, epoch)

