# %%
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from vessel_proj.preprocess_data.gnn import get_graph_from_saved_data, get_train_mask
import mlflow
from vessel_proj.ds_utils import set_mlflow
from vessel_proj.ds_utils.torch.opt import optimize_torch_obj_fun


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

# Get Experiment Details
experiment = mlflow.get_experiment_by_name(experiment_name)

with mlflow.start_run(experiment_id=experiment.experiment_id):

    graph = get_graph_from_saved_data()
    graph.num_classes = graph.y.unique().shape[0]
    graph.x
    n = 1000

    graph.train_mask, graph.test_mask = get_train_mask(graph.x.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        print(f"Accuracy: {acc:.4f}")
# %%
# %%
with mlflow.start_run(run_id=parent_run.info.run_id, nested=True):
    with mlflow.start_run(experiment_id=parent_run.info.experiment_id, nested=True):

        logger.info(run_par_dict)

        # save files in temp folder, then log them as artifacts in mlflow and delete temp fold

        with tempfile.TemporaryDirectory() as tmpdirname:

            # set artifacts folders and subfolders
            tmp_path = Path(tmpdirname)
            dgp_fold = tmp_path / "dgp"
            dgp_fold.mkdir(exist_ok=True)
            tb_fold = tmp_path / "tb_logs"
            tb_fold.mkdir(exist_ok=True)

            mlflow.log_params(parent_runs_par)
            bin_or_w, mod_dgp = list(mod_dgp_dict.items())[0]
            for bin_or_w, mod_dgp in mod_dgp_dict.items():
                mlflow.log_params(
                    {
                        f"dgp_{bin_or_w}_{key}": val
                        for key, val in run_par_dict[f"dgp_par_{bin_or_w}"].items()
                    }
                )
                mlflow.log_params(
                    {
                        f"filt_{bin_or_w}_{key}": val
                        for key, val in run_par_dict[f"filt_par_{bin_or_w}"].items()
                    }
                )
                logger.info(f" start estimates {bin_or_w}")
                if parent_runs_par["use_lag_mat_as_reg"]:
                    if mod_dgp.X_T.shape[2] != 1:
                        raise Exception(" multiple lags not ready yet")
                    logger.info("Using lagged adjacency matrix as regressor")
                    use_lag_mat_as_reg = True
                else:
                    use_lag_mat_as_reg = False

                # sample obs from dgp and save data
                if hasattr(mod_dgp, "bin_mod"):
                    if mod_dgp.bin_mod.Y_T.sum() == 0:
                        mod_dgp.bin_mod.sample_and_set_Y_T(
                            use_lag_mat_as_reg=use_lag_mat_as_reg
                        )

                    mod_dgp.sample_and_set_Y_T(
                        A_T=mod_dgp.bin_mod.Y_T,
                        use_lag_mat_as_reg=use_lag_mat_as_reg,
                    )
                else:
                    mod_dgp.sample_and_set_Y_T(use_lag_mat_as_reg=use_lag_mat_as_reg)

                torch.save(run_data_dict["Y_reference"], dgp_fold / "Y_reference.pt")
                torch.save(
                    (mod_dgp.get_Y_T_to_save(), mod_dgp.X_T),
                    dgp_fold / "obs_T_dgp.pt",
                )
                mod_dgp.save_parameters(save_path=dgp_fold)
                if mod_dgp.phi_T is not None:
                    mlflow.log_figure(
                        mod_dgp.plot_phi_T()[0], f"fig/{bin_or_w}_dgp_all.png"
                    )

                # estimate models and log parameters and hpar optimization
                filt_models = get_filt_mod(
                    bin_or_w,
                    mod_dgp.Y_T,
                    mod_dgp.X_T,
                    run_par_dict[f"filt_par_{bin_or_w}"],
                )

                # k_filt, mod = list(filt_models.items())[0]
                for k_filt, mod_filt in filt_models.items():

                    _, h_par_opt, stats_opt = mod_filt.estimate(tb_save_fold=tb_fold)

                    mlflow.log_params(
                        {
                            f"filt_{bin_or_w}_{k_filt}_{key}": val
                            for key, val in h_par_opt.items()
                        }
                    )
                    mlflow.log_metrics(
                        {
                            f"filt_{bin_or_w}_{k_filt}_{key}": val
                            for key, val in stats_opt.items()
                        }
                    )
                    mlflow.log_params(
                        {
                            f"filt_{bin_or_w}_{k_filt}_{key}": val
                            for key, val in mod_filt.get_info_dict().items()
                            if key not in h_par_opt.keys()
                        }
                    )

                    mod_filt.save_parameters(save_path=tmp_path)

                    # compute mse for each model and log it
                    nodes_to_exclude = mod_dgp.get_inds_inactive_nodes()

                    mse_dict = filt_err(
                        mod_dgp, mod_filt, suffix=k_filt, prefix=bin_or_w
                    )
                    mlflow.log_metrics(mse_dict)
                    logger.warning(mse_dict)

                    # mlflow.log_metrics({f"filt_{bin_or_w}_{k_filt}_{key}": v for key, v in mod_filt.out_of_sample_eval().items()})

                    # log plots that can be useful for quick visual diagnostic
                    if mod_filt.phi_T is not None:
                        mlflow.log_figure(
                            mod_filt.plot_phi_T()[0],
                            f"fig/{bin_or_w}_{k_filt}_filt_all.png",
                        )
                    i_plot = torch.where(~splitVec(nodes_to_exclude)[0])[0][0]

                    if mod_filt.phi_T is not None:
                        if mod_dgp.phi_T is not None:
                            fig_ax = mod_dgp.plot_phi_T(i=i_plot)
                        else:
                            fig_ax = None

                        mlflow.log_figure(
                            mod_filt.plot_phi_T(i=i_plot, fig_ax=fig_ax)[0],
                            f"fig/{bin_or_w}_{k_filt}_filt_phi_ind_{i_plot}.png",
                        )

                    if mod_dgp.X_T is not None:
                        avg_beta_dict = {
                            f"{bin_or_w}_{k}_dgp": v
                            for k, v in mod_dgp.get_avg_beta_dict().items()
                        }
                        mlflow.log_metrics(avg_beta_dict)
                        mlflow.log_metric(
                            f"{bin_or_w}_avg_beta_dgp",
                            np.mean(list(avg_beta_dict.values())),
                        )

                        fig = plt.figure()
                        plt.plot(mod_dgp.X_T[0, 0, :, :].T, figure=fig)
                        mlflow.log_figure(fig, f"fig/{bin_or_w}_X_0_0_T.png")

                    if mod_dgp.any_beta_tv():
                        plot_dgp_fig_ax = mod_dgp.plot_beta_T()
                        mlflow.log_figure(
                            plot_dgp_fig_ax[0], f"fig/{bin_or_w}_sd_filt_beta_T.png"
                        )
                    if mod_filt.beta_T is not None:
                        if mod_filt.any_beta_tv():
                            mlflow.log_figure(
                                mod_filt.plot_beta_T(fig_ax=plot_dgp_fig_ax)[0],
                                f"fig/{bin_or_w}_{k_filt}_filt_beta_T.png",
                            )

            # log all files and sub-folders in temp fold as artifacts
            mlflow.log_artifacts(tmp_path)
