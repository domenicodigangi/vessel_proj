import torch
import torch_geometric
import pandas as pd
from vessel_proj.preprocess_data.get_and_save_centralities_from_graph_data import (
    get_graph_data,
)
import numpy as np
from sklearn.model_selection import train_test_split
from vessel_proj.task.classification_task_pipeline_ports_centr import (
    encode_df_features,
    encode_features,
    simple_impute_cols,
)
from vessel_proj.preprocess_data import (
    get_latest_port_data_task,
    get_latest_port_features,
)


def get_graph_links(vessel_category):
    """get links data"""
    idx = pd.IndexSlice
    trips_all = get_graph_data()
    df_links = trips_all.loc[idx[:, :, vessel_category]]
    return df_links


def get_node_features():
    df_features = get_latest_port_features()

    feat_names_non_cat = ["TIDE_RANGE", "LATITUDE", "LONGITUDE"]

    feat_names = list(df_features.columns)
    feat_names_cat = [f for f in feat_names if f not in feat_names_non_cat]

    df_features_enc = encode_df_features(df_features, feat_names_cat)

    feat_imputed, _ = simple_impute_cols(feat_names_non_cat, df_features_enc)

    return feat_imputed


def combine_feat_and_links(df_links, feat):
    """clean indices as integers from 1 to N using the ordering of the features df"""
    clean_inds = (
        feat.reset_index()["INDEX_NO"]
        .reset_index()
        .rename(columns={"index": "new_ind", "INDEX_NO": "old_ind"})
    )

    trips_new_inds = (
        df_links.reset_index()
        .merge(
            clean_inds,
            how="inner",
            left_on="start_port",
            right_on="old_ind",
            validate="many_to_one",
        )
        .rename(columns={"new_ind": "start_port_new_ind"})
        .merge(
            clean_inds,
            how="inner",
            left_on="end_port",
            right_on="old_ind",
            validate="many_to_one",
        )
        .rename(columns={"new_ind": "end_port_new_ind"})
        .drop(columns=["old_ind_x", "old_ind_y", "start_port", "end_port"])
    )

    edges = torch.tensor(
        trips_new_inds[["start_port_new_ind", "end_port_new_ind"]].to_numpy().T,
        dtype=torch.long,
    )

    edges_attr = torch.tensor(trips_new_inds[["trips_count"]].to_numpy().T).float()

    return edges, edges_attr, feat


def get_graph(target_feat: str, edges, edges_attr, feat) -> torch_geometric.data.Data:
    # Convert the graph information into a PyG Data object

    y = torch.tensor(feat[target_feat].to_numpy()).long()
    feat = torch.tensor(feat.drop(columns=target_feat).to_numpy()).float()
    graph = torch_geometric.data.Data(
        x=feat, edge_index=edges, edge_attr=edges_attr, y=y
    )

    return graph


def get_graph_from_saved_data(target_feat="COUNTRY", vessel_category="cargo"):
    df_links = get_graph_links(vessel_category)
    feat_nodes = get_node_features()
    edges, edges_attr, feat = combine_feat_and_links(df_links, feat_nodes)

    graph = get_graph(target_feat, edges, edges_attr, feat)
    return graph


def get_train_mask(n, train_size=0.7, seed=1):

    tot_inds = range(n)
    inds_train, inds_val_test = train_test_split(
        tot_inds, train_size=train_size, random_state=seed
    )

    inds_val, inds_test = train_test_split(
        inds_val_test, test_size=0.5, random_state=seed
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[inds_train] = True
    val_mask[inds_val] = True
    test_mask[inds_test] = True

    # train_mask & val_mask
    # train_mask & test_mask
    # val_mask & test_mask

    return train_mask, val_mask, test_mask
