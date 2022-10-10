#%%
from tracemalloc import get_traceback_limit
from typing import List, Tuple
import pandas as pd
from geopy import distance
from vessel_proj.ds_utils import get_data_path
import torch
import torch_geometric
import sklearn
import numpy as np
from sklearn.neighbors import BallTree
import networkx as nx
from pathlib import Path

# TODO add category
# TODO save in parquet
# TODO run function on all files


#%%


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


def get_list_df_traj_from_df_full_traj(df_full_traj: pd.DataFrame):

    df_full_traj = df_full_traj.drop(columns=["na", "date_str", "time_str"])
    df_full_traj = df_full_traj.drop_duplicates()
    df_full_traj = df_full_traj.sort_values(by="date_float")

    n_min_points_per_traj = 100
    n_5_min_intervals_in_one_day = 288
    n_seconds_in_one_day = 60 * 60 * 24
    max_delta = 1 / n_5_min_intervals_in_one_day
    df_full_traj["traj_group"] = (
        df_full_traj["date_float"].diff() > max_delta
    ).cumsum()
    df_full_traj["traj_group"].iloc[0] = df_full_traj["traj_group"].iloc[1]
    df_full_traj["lat_long"] = df_full_traj[["latitude", "longitude"]].apply(
        tuple, axis=1
    )

    df_full_traj["lat_long_prev"] = df_full_traj["lat_long"].shift(-1)
    df_full_traj["one_step_distance_meters"] = (
        df_full_traj[["lat_long", "lat_long_prev"]]
        .dropna()
        .apply(lambda x: distance.distance(x["lat_long"], x["lat_long_prev"]).m, axis=1)
    )

    df_full_traj["delta_date_float"] = (
        df_full_traj["date_float"].shift(-1) - df_full_traj["date_float"]
    )
    df_full_traj["delta_date_seconds"] = (
        df_full_traj["delta_date_float"] * n_seconds_in_one_day
    )
    df_full_traj["speed_m_s"] = (
        df_full_traj["one_step_distance_meters"] / df_full_traj["delta_date_seconds"]
    )

    df_traj_list = list(
        filter(
            lambda x: x[1].shape[0] > n_min_points_per_traj,
            df_full_traj.groupby("traj_group"),
        )
    )

    return df_traj_list


def get_chain_graph_from_sub_traj(df_traj: pd.DataFrame):

    chain_array_of_neighbours = get_chain_links(df_traj)

    df_edges = get_df_edge_list_from_array_of_neighbours(chain_array_of_neighbours)
    df_feat = df_traj[
        [
            "latitude",
            "longitude",
            "altitude",
            "one_step_distance_meters",
            "delta_date_seconds",
            "speed_m_s",
        ]
    ]

    graph = get_traj_graph(df_edges, df_feat)

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
    df_edges: pd.DataFrame, df_feat: pd.DataFrame
) -> torch_geometric.data.Data:

    edges = torch.tensor(
        df_edges[["start_node", "end_node"]].to_numpy().T,
        dtype=torch.long,
    )

    feat = torch.tensor(df_feat.to_numpy()).float()
    graph = torch_geometric.data.Data(x=feat, edge_index=edges)

    return graph


#%%

load_path = get_data_path() / "raw" / "Geolife Trajectories 1.3" / "Data"
foldpath = Path("/home/digan/cnr/vessel_proj/data/raw/Geolife Trajectories 1.3/Data/")


for subfold in foldpath.iterdir():

    df_labels = pd.read_csv(subfold / "labels.txt")

    traj_subfold = subfold / "Trajectory"

    for i, filepath in enumerate(traj_subfold.iterdir()):
        row = df_labels.iloc[i, :]
        mode = row["mode"]

        df_full_traj = pd.read_csv(
            filepath,
            skiprows=6,
            header=None,
            names=[
                "latitude",
                "longitude",
                "na",
                "altitude",
                "date_float",
                "date_str",
                "time_str",
            ],
        )

        # TODO check that label corresponds to traj

        df_traj_list = get_list_df_traj_from_df_full_traj(df_full_traj)

        # TODO add mode to df and save parquet with subtraj in unique large folder

df_traj_list = get_list_df_traj_from_df_full_traj(filepath)

graph_list = []

for g, df_traj in df_traj_list:
    graph = get_chain_graph_from_sub_traj(df_traj)
    graph_list.append(graph)


#%% Visual


def visualize_graphs(df_traj):
    altitude_array_of_neighbours, _ = get_altitude_links(df_traj, th_meters=5)
    horizontal_array_of_neighbours, _ = get_horiz_links(df_traj, th_meters=30)
    time_array_of_neighbours, _ = get_temporal_links(df_traj, th_seconds=10)
    chain_array_of_neighbours = get_chain_links(df_traj)

    G_chain = nx.from_edgelist(get_edge_list(chain_array_of_neighbours))
    G_time = nx.from_edgelist(get_edge_list(time_array_of_neighbours))
    G_time.remove_edges_from(nx.selfloop_edges(G_time))
    G_altitude = nx.from_edgelist(get_edge_list(altitude_array_of_neighbours))
    G_altitude.remove_edges_from(nx.selfloop_edges(G_altitude))
    G_horizontal_dist = nx.from_edgelist(get_edge_list(horizontal_array_of_neighbours))
    G_horizontal_dist.remove_edges_from(nx.selfloop_edges(G_horizontal_dist))

    alpha = 0.3
    nx.draw(
        G_chain,
        pos=df_traj[["lat_long"]].to_dict()["lat_long"],
        node_size=10,
        alpha=alpha,
    )
    nx.draw(
        G_time,
        pos=df_traj[["lat_long"]].to_dict()["lat_long"],
        node_size=10,
        alpha=alpha,
    )
    nx.draw(
        G_altitude,
        pos=df_traj[["lat_long"]].to_dict()["lat_long"],
        node_size=10,
        alpha=alpha,
    )
    nx.draw(
        G_horizontal_dist,
        pos=df_traj[["lat_long"]].to_dict()["lat_long"],
        node_size=10,
        alpha=alpha,
    )


# %%
