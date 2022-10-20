from typing import List, Tuple
import pandas as pd
import torch
import torch_geometric
import numpy as np
from sklearn.neighbors import BallTree
from pathlib import Path


def get_horiz_links(df: pd.DataFrame, th_meters: int) -> Tuple:

    hearth_radius = 6371000

    X = df[["latitude", "longitude"]]
    tree = BallTree(X, metric="haversine")
    th_angle = th_meters / hearth_radius
    res = tree.query_radius(X, th_angle, return_distance=True)

    array_of_neighbours = res[0]
    dist_in_meters = res[1] * hearth_radius

    return array_of_neighbours, dist_in_meters


def get_temporal_links(df: pd.DataFrame, th_seconds: int) -> Tuple:

    n_seconds_in_one_day = 60 * 60 * 24

    delta_secs_since_first = (
        df["date_float"] - df["date_float"][0]
    ) * n_seconds_in_one_day

    X = np.expand_dims(delta_secs_since_first.values, axis=1)

    tree = BallTree(X, metric="euclidean")
    res = tree.query_radius(X, th_seconds, return_distance=True)

    return res


def get_edge_list(array_of_neighbours: np.array) -> List[Tuple[int, int]]:

    edge_list = []
    for i, arr in enumerate(array_of_neighbours):
        list_edges_i = [(i, neig_ind) for neig_ind in arr]  # if i != neig_ind]
        edge_list.extend(list_edges_i)

    return edge_list


def get_altitude_links(df: pd.DataFrame, th_meters: int) -> Tuple:

    X = df[["altitude"]]

    tree = BallTree(X, metric="euclidean")
    res = tree.query_radius(X, th_meters, return_distance=True)

    return res


def get_chain_links(df: pd.DataFrame) -> Tuple:

    res = np.array([np.array([i + 1]) for i in range(df.shape[0] - 1)])

    return res


def get_chain_graph_from_sub_traj(
    df_traj: pd.DataFrame,
    cat_col: str,
    additional_columns: List = [],
):

    chain_array_of_neighbours = get_chain_links(df_traj)

    df_edges = get_df_edge_list_from_array_of_neighbours(chain_array_of_neighbours)
    df_feat = df_traj[
        [
            "latitude",
            "longitude",
            "one_step_distance_meters",
            "delta_date_seconds",
            "speed_m_s",
        ]
        + additional_columns
    ]

    df_feat = df_feat.fillna(method="ffill")

    if df_feat.isnull().any().any():
        print("null value found")
        df_feat = df_feat.dropna()
    if df_feat.isin([np.inf, -np.inf]).any().any():
        raise ValueError("inf value found")

    if df_traj[cat_col].unique().shape[0] != 1:
        raise ValueError("should have only one category per trajectory")
    transportation_mode = df_traj[cat_col].iloc[0]

    graph = get_traj_graph(df_edges, df_feat, transportation_mode)

    return graph


def get_df_edge_list_from_array_of_neighbours(
    array_of_neighbours: np.array,
) -> pd.DataFrame:

    df_edges = (
        pd.DataFrame(array_of_neighbours)
        .explode(0)
        .reset_index()
        .rename(columns={"index": "start_node", 0: "end_node"})
    )
    return df_edges


def get_traj_graph(
    df_edges: pd.DataFrame, df_feat: pd.DataFrame, y: str
) -> torch_geometric.data.Data:

    edges = torch.tensor(
        df_edges[["start_node", "end_node"]].to_numpy().T,
        dtype=torch.long,
    )

    feat = torch.tensor(df_feat.to_numpy()).float()
    graph = torch_geometric.data.Data(x=feat, edge_index=edges, y=y)

    return graph
