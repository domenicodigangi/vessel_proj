import copy
import io
import logging
import tempfile
from pathlib import Path
from typing import List

import mlflow
import torch
from torch.utils.tensorboard import SummaryWriter
from vessel_proj.ds_utils.torch.opt import grad_norm_from_list

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train(model, train_loader, optimizer, criterion):
    model.train()

    is_heterogeneous = check_if_heterogeneous(model)

    for data in train_loader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.
        if is_heterogeneous:
            out_dict = model(
                data.x_dict, data.edge_index_dict, data.batch_dict
            )  # Perform a single forward pass.
            loss = torch.stack(
                [criterion(out, data.y) for node_type, out in out_dict.items()]
            ).sum()  # Compute the loss.
        else:
            out = model(data.x, data.edge_index, data.batch)

            loss = criterion(out, data.y)  # Compute the loss.

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        # print(loss.data)

    return loss


def get_accuracy(loader, model):
    model.eval()
    is_heterogeneous = check_if_heterogeneous(model)

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        if is_heterogeneous:
            out_dict = model(
                data.x_dict, data.edge_index_dict, data.batch_dict
            )  # Perform a single forward pass.
            pred = out_dict["traj_point"].argmax(
                dim=1
            )  # Use the class with highest probability.
        else:
            out = model(data.x, data.edge_index, data.batch)

            pred = out.argmax(dim=1)  # Use the class with highest probability.

        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def check_if_heterogeneous(model):
    if "Hetero" in model.__str__():
        is_heterogeneous = True
    else:
        is_heterogeneous = False
    return is_heterogeneous


def get_x_from_homo_and_hetero_graph(graph):

    try:
        x = graph.x
    except AttributeError:
        x = graph["traj_point"].x
    return x


def execute_one_run(
    experiment,
    model,
    train_loader,
    val_loader,
    lr_values: List[float],
    graph_type,
    n_epochs: int,
    h_par_init: dict = {},
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    for lr in lr_values:

        with mlflow.start_run(experiment_id=experiment.experiment_id):
            model = model.to(device)
            torch.manual_seed(12345)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            h_par = h_par_init

            for k, v in optimizer.param_groups[0].items():
                if k not in "params":
                    h_par[k] = v

            h_par["optim_str"] = optimizer.__str__()
            h_par["n_epochs_max"] = n_epochs
            h_par["n_layers"] = len(model.convs)
            h_par["Model Name"] = model._get_name()
            h_par["hidden_channels"] = model.hidden_channels
            h_par["Model String"] = model.__str__()
            h_par["graph_type"] = graph_type
            h_par["early_stopping_patience"] = 50
            h_par["early_stopping_min_delta"] = 0

            print(h_par)
            mlflow.log_params(h_par)

            with tempfile.TemporaryDirectory() as tmpdirname:

                tmp_path = Path(tmpdirname)
                tb_fold = tmp_path / "tb_logs"
                tb_fold.mkdir(exist_ok=True)
                logger.info(f"tensorboard logs in {tb_fold}")
                checkpoints_fold = tmp_path / "checkpoints"
                checkpoints_fold.mkdir(exist_ok=True)
                writer = SummaryWriter(str(tb_fold))

                es = EarlyStopping(
                    patience=h_par["early_stopping_patience"],
                    min_delta=h_par["early_stopping_min_delta"],
                )
                # log all files and sub-folders in temp fold as artifacts
                epoch = 0
                done = False
                while epoch < h_par["n_epochs_max"] and not done:
                    epoch += 1

                    loss = train(model, train_loader, optimizer, criterion)

                    train_accuracy = get_accuracy(train_loader, model)
                    valid_accuracy = get_accuracy(val_loader, model)
                    print(
                        f"Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_accuracy:.4f}, Validation Acc: {valid_accuracy:.4f}  no improvement: {es.counter}"
                    )

                    writer.add_scalar("Loss/value", loss.item(), epoch)
                    writer.add_scalar("Valid Accuracy", valid_accuracy, epoch)
                    writer.add_scalar("Train Accuracy", train_accuracy, epoch)

                    grad_norm = grad_norm_from_list(model.parameters())
                    writer.add_scalar("Loss/grad_norm", grad_norm, epoch)

                    if es(model, valid_accuracy, loss):
                        print(es.status)
                        done = True

                logger.info(f"Saving final artifacts at epoch {epoch}")
                filepath_checkpoint = checkpoints_fold / f"checkpoint_{epoch}.pt"
                torch.save(
                    {
                        "model_state_dict": es.best_model.state_dict(),
                    },
                    filepath_checkpoint,
                )

                mlflow.log_artifact(filepath_checkpoint)

                mlflow.log_metrics(
                    {
                        "Loss/value": es.best_train_loss.item(),
                        "Valid Accuracy": es.best_validation_accuracy,
                        "n_epochs": epoch,
                    }
                )

                mlflow.log_artifacts(tmp_path)

    mlflow.end_run()


class EarlyStopping:
    def __init__(self, patience=25, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_validation_accuracy = None
        self.best_train_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_accuracy, train_loss):
        if self.best_validation_accuracy == None:
            self.best_validation_accuracy = val_accuracy
            self.best_train_loss = train_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_validation_accuracy - val_accuracy < self.min_delta:
            self.best_validation_accuracy = val_accuracy
            self.best_train_loss = train_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_validation_accuracy - val_accuracy > self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = f"Stopped on {self.counter} with best train loss {self.best_train_loss} and best valid accuracy {self.best_validation_accuracy}"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        self.status = f"{self.counter}/{self.patience}"
        return False
