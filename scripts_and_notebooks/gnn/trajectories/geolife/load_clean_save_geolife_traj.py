#%%
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
import datetime

#%%


def get_list_df_traj_from_df_full_traj(
    df_full_traj: pd.DataFrame, df_labels: pd.DataFrame
) -> List[pd.DataFrame]:

    df_full_traj["datetime"] = pd.to_datetime(
        df_full_traj["date_float"], unit="D", origin=origin
    )

    df_full_traj = df_full_traj.sort_values(by="datetime")
    df_full_traj["datetime_2"] = pd.to_datetime(
        df_full_traj["date_str"] + " " + df_full_traj["time_str"]
    )
    ind_same_start = df_full_traj["datetime_2"].iloc[0] == df_labels["Start Time"]
    ind_same_end = df_full_traj["datetime_2"].iloc[-1] == df_labels["End Time"]
    ind_same = ind_same_end & ind_same_start
    if ind_same.sum() == 1:
        df_full_traj["transportation_mode"] = df_labels["Transportation Mode"][
            ind_same
        ].iloc[0]

        df_full_traj = df_full_traj.drop(
            columns=["na", "date_str", "time_str", "date_float"]
        )
        df_full_traj = df_full_traj.drop_duplicates()
        df_full_traj = df_full_traj.sort_values(by="datetime")

        n_min_points_per_traj = 100
        max_delta = datetime.timedelta(minutes=5)

        df_full_traj["traj_group"] = (
            df_full_traj["datetime"].diff() > max_delta
        ).cumsum()
        df_full_traj["traj_group"].iloc[0] = df_full_traj["traj_group"].iloc[1]
        df_full_traj["lat_long"] = df_full_traj[["latitude", "longitude"]].apply(
            tuple, axis=1
        )

        df_full_traj = df_full_traj.drop(
            df_full_traj[df_full_traj["latitude"] > 90].index
        )
        df_full_traj = df_full_traj.drop(
            df_full_traj[df_full_traj["latitude"] < -90].index
        )

        df_full_traj["lat_long_prev"] = df_full_traj["lat_long"].shift(1)
        df_full_traj["one_step_distance_meters"] = (
            df_full_traj[["lat_long", "lat_long_prev"]]
            .dropna()
            .apply(
                lambda x: distance.distance(x["lat_long"], x["lat_long_prev"]).m, axis=1
            )
        )

        df_full_traj["delta_datetime"] = (
            df_full_traj["datetime"].shift(-1) - df_full_traj["datetime"]
        )
        df_full_traj["delta_date_seconds"] = df_full_traj["delta_datetime"].dt.seconds
        df_full_traj["speed_m_s"] = (
            df_full_traj["one_step_distance_meters"]
            / df_full_traj["delta_date_seconds"]
        )

        df_full_traj["datetime"] = df_full_traj["datetime"].astype("datetime64[ms]")

        df_traj_list = [
            df_sub_traj
            for i, df_sub_traj in df_full_traj.groupby("traj_group")
            if df_sub_traj.shape[0] > n_min_points_per_traj
        ]

        if df_full_traj["speed_m_s"].isin([np.inf, -np.inf]).any():
            print("Infinite speed found, skipping")
            df_traj_list = []
    else:
        df_traj_list = []

    return df_traj_list


#%%

load_path = get_data_path() / "raw" / "Geolife Trajectories 1.3" / "Data"
foldpath = Path("/home/digan/cnr/vessel_proj/data/raw/Geolife Trajectories 1.3/Data/")
savefoldpath = Path(
    "/home/digan/cnr/vessel_proj/data/interim/geolife_traj_preprocessed"
)
savefoldpath.mkdir(exist_ok=True)

filepath_labels_dict = {}
for subfold in foldpath.iterdir():
    label_filepath = subfold / "labels.txt"
    if label_filepath.exists():
        filepath_labels_dict[subfold.name] = label_filepath

for traj_id, label_filepath in filepath_labels_dict.items():
    df_labels = pd.read_csv(label_filepath, sep="\t")

    traj_subfold = label_filepath.parent / "Trajectory"
    df_labels["Start Time"] = pd.to_datetime(df_labels["Start Time"])
    df_labels["End Time"] = pd.to_datetime(df_labels["End Time"])

    origin = datetime.date.fromisoformat("1899-12-30")
    for i, filepath in enumerate(traj_subfold.iterdir()):
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

        df_traj_list = get_list_df_traj_from_df_full_traj(df_full_traj, df_labels)

        for j, df_traj in enumerate(df_traj_list):
            savefilepath = savefoldpath / f"{traj_id}_{i}_{j}.parquet"
            df_traj.to_parquet(savefilepath)


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
