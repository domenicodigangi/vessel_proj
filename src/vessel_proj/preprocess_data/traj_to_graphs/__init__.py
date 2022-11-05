from dis import dis
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
import torch_geometric
import numpy as np
from sklearn.neighbors import BallTree
from pathlib import Path


def get_temporal_links(df: pd.DataFrame, th_seconds: int) -> Tuple:

    n_seconds_in_one_day = 60 * 60 * 24

    delta_secs_since_first = (
        df["date_float"] - df["date_float"][0]
    ) * n_seconds_in_one_day

    X = np.expand_dims(delta_secs_since_first.values, axis=1)

    tree = BallTree(X, metric="euclidean")
    res = tree.query_radius(X, th_seconds, return_distance=True)

    return res


def get_neighbours_in_radius(X: np.ndarray, radius: int, metric="euclidean") -> Tuple:

    tree = BallTree(X, metric=metric)
    dist, ind = tree.query_radius(X, radius, return_distance=True)

    return dist, ind


def get_k_nearest_neighbours(X: np.ndarray, k: int, metric: str = "euclidean") -> Tuple:

    tree = BallTree(X, metric=metric)
    dist, ind = tree.query(X, k=k)

    return dist, ind


def get_edge_list(array_of_neighbours: np.array) -> List[Tuple[int, int]]:

    edge_list = []
    for i, arr in enumerate(array_of_neighbours):
        list_edges_i = [(i, neig_ind) for neig_ind in arr]  # if i != neig_ind]
        edge_list.extend(list_edges_i)

    return edge_list


def get_chain_graph_from_traj(
    df_traj: pd.DataFrame,
    cat_col: str,
    additional_columns: List = [],
):

    chain_array_of_neighbours = get_temporal_chain_links(df_traj)

    df_edges = get_df_edge_list_from_uniform_array_of_neighbours(
        chain_array_of_neighbours
    )

    df_feat = get_df_feat_from_traj(df_traj, additional_columns=additional_columns)

    if df_traj[cat_col].unique().shape[0] != 1:
        raise ValueError("should have only one category per trajectory")
    y_value = df_traj[cat_col].iloc[0]

    graph = get_traj_graph(df_edges, df_feat, y_value)

    return graph


def get_temporal_chain_links(df_traj: pd.DataFrame) -> Tuple:

    res = np.array([np.array([i + 1]) for i in range(df_traj.shape[0] - 1)])

    return res


def get_lat_long_neighbours(df_traj: pd.DataFrame, method: str, par: float) -> Tuple:

    hearth_radius = 6371000

    X = df_traj[["latitude", "longitude"]]

    if method == "knn":
        k = par
        dist, ind = get_k_nearest_neighbours(X, k=k, metric="haversine")

    elif method == "radius":
        radius_meters = par
        th_angle = radius_meters / hearth_radius

        dist, ind = get_neighbours_in_radius(X, radius=th_angle, metric="haversine")

    else:
        raise ValueError("method not recognized")

    dist_hor = dist * hearth_radius

    return dist_hor, ind


def get_df_feat_from_traj(
    df_traj: pd.DataFrame, additional_columns: List = []
) -> pd.DataFrame:

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

    return df_feat


def get_df_edge_list_from_uniform_array_of_neighbours(
    array_of_neighbours: np.array,
    distances: Optional[np.array] = None,
) -> pd.DataFrame:

    df_edges = (
        pd.DataFrame(array_of_neighbours)
        .stack()
        .reset_index()[["level_0", 0]]
        .rename(columns={"level_0": "start_node", 0: "end_node"})
    )
    if distances is not None:
        df_edges["distance"] = pd.DataFrame(distances).stack().reset_index()[0]

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


def get_homogeneous_spatio_temp_chain_graph(
    df_traj: pd.DataFrame,
    cat_col: str,
    additional_columns: List = [],
) -> torch_geometric.data.Data:

    dict_df_edges = get_spatio_temp_edges_dict(df_traj)

    df = pd.concat(dict_df_edges.values())

    df_edges = remove_self_loops_and_dup(df)

    df_feat = get_df_feat_from_traj(df_traj, additional_columns=additional_columns)

    y_value = df_traj[cat_col].iloc[0]

    graph = get_traj_graph(df_edges, df_feat, y_value)

    return graph


def get_spatio_temp_edges_dict(
    df_traj: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:

    k = 2
    dist_hor, edges_hor = get_lat_long_neighbours(df_traj, method="knn", par=k)

    edges_temp = get_temporal_chain_links(df_traj)

    df_edges_temp = get_df_edge_list_from_uniform_array_of_neighbours(edges_temp)
    df_edges_hor = get_df_edge_list_from_uniform_array_of_neighbours(edges_hor)

    dict_df_edges = {"temporal": df_edges_temp, "horizontal": df_edges_hor}

    if "altitude" in df_traj.columns:
        dist, edges_vert = get_k_nearest_neighbours(
            np.expand_dims(df_traj["altitude"].values, axis=1), k=k, metric="euclidean"
        )
        df_edges_vert = get_df_edge_list_from_uniform_array_of_neighbours(edges_vert)

        dict_df_edges["altitude"] = df_edges_vert

    return dict_df_edges


def remove_self_loops_and_dup(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["start_node"] != df["end_node"]].drop_duplicates()


def clean_and_keep_closest_k_links(df: pd.DataFrame, n_to_keep: int) -> pd.DataFrame:
    df = remove_self_loops_and_dup(df)

    return df.sort_values(by="distance").iloc[:n_to_keep]


def get_homogeneous_spatio_temp_closest_percentile_graph(
    df_traj: pd.DataFrame,
    cat_col: str,
    percentile: float = 1.0,
    additional_columns: List = [],
) -> torch_geometric.data.Data:

    n_nodes = df_traj.shape[0]
    n_pox_links = n_nodes * (n_nodes - 1) / 2
    n_links_percentile = round(n_pox_links * (percentile / 100)) + 1
    k = round((n_links_percentile / n_nodes) * 10)

    dist_hor, edges_hor = get_lat_long_neighbours(df_traj, method="knn", par=k)

    df_edges_hor = get_df_edge_list_from_uniform_array_of_neighbours(
        edges_hor, dist_hor
    )

    df_edges_hor = clean_and_keep_closest_k_links(df_edges_hor, n_links_percentile)

    dist_temp, edges_temp = get_k_nearest_neighbours(
        np.expand_dims(
            (df_traj["datetime"] - df_traj["datetime"].iloc[0]).values, axis=1
        ),
        k=k,
        metric="euclidean",
    )
    df_edges_temp = get_df_edge_list_from_uniform_array_of_neighbours(
        edges_temp, dist_temp
    )
    df_edges_temp = clean_and_keep_closest_k_links(df_edges_temp, n_links_percentile)

    df = pd.concat((df_edges_temp, df_edges_hor))

    if "altitude" in df_traj.columns:
        dist_vert, edges_vert = get_k_nearest_neighbours(
            np.expand_dims(df_traj["altitude"].values, axis=1),
            k=k,
            metric="euclidean",
        )
        df_edges_vert = get_df_edge_list_from_uniform_array_of_neighbours(
            edges_vert, dist_vert
        )
        df_edges_vert = clean_and_keep_closest_k_links(
            df_edges_temp, n_links_percentile
        )

    df = pd.concat((df, df_edges_vert))

    df_edges = remove_self_loops_and_dup(df)

    df_feat = get_df_feat_from_traj(df_traj, additional_columns=additional_columns)

    y_value = df_traj[cat_col].iloc[0]

    graph = get_traj_graph(df_edges, df_feat, y_value)

    return graph


def get_het_graph_from_df_traj_and_df_feat(
    df_traj: pd.DataFrame,
    cat_col: str,
    additional_columns: List = [],
) -> torch_geometric.data.Data:

    dict_df_edges = get_spatio_temp_edges_dict(df_traj)

    df_feat = get_df_feat_from_traj(df_traj, additional_columns=additional_columns)

    feat = torch.tensor(df_feat.to_numpy()).float()

    y_value = df_traj[cat_col].iloc[0]

    het_graph = torch_geometric.data.HeteroData()
    het_graph["traj_point"].x = feat  # [num_papers, num_features_paper]

    for rel_name, df_edges in dict_df_edges.items():

        edges = torch.tensor(
            df_edges[["start_node", "end_node"]].to_numpy().T,
            dtype=torch.long,
        )

        het_graph["traj_point", rel_name, "traj_point"].edge_index = edges

        het_graph.y = y_value

    return het_graph
