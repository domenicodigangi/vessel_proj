#%%
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from vessel_proj.preprocess_data import get_data_path, set_types_edge_list, get_wandb_root_path, get_project_name
from prefect import task


import logging

logger = logging.getLogger("root")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

from prefect import task, flow

# %%



def clean_edges(df_edges: pd.DataFrame, min_dur_secs=300) -> pd.DataFrame:
    """
    Remove weird links
    """
    # Too short trip duration
    ind_no_dur = df_edges["duration_seconds"] < min_dur_secs
    logger.info(f"found {ind_no_dur.sum() } links with zero duration")

    # same source and dest
    ind_same_ports = df_edges["start_port"] == df_edges["end_port"]
    logger.info(f"found {ind_same_ports.sum() } links from the same port")

    # drop them
    ind_to_drop = ind_no_dur | ind_same_ports
    logger.info(f"dropping  {ind_to_drop.sum() } links ")
    df_edges.drop(df_edges[ind_to_drop].index, inplace=True)

    return df_edges


def group_edges_per_vesseltype(df_edges: pd.DataFrame) -> pd.DataFrame:
    #%% group trips
    # count number of connections between each pairs of ports, avg duration, number of distinct vessels and types of vessels

    count_unique = lambda x: np.unique(x).shape[0]
    df_edges_per_vesseltype = df_edges.groupby(["start_port", "end_port", "vesseltype"], observed=True).agg(
        duration_avg_days=("duration_days", np.mean),
        trips_count=("uid", count_unique),
    )

    return df_edges_per_vesseltype


#%%

@task
def main(min_dur_secs=300):
    """load list of voyages (edge_list), clean the graph, compute a set of centralities and log them as parquet"""

    #%% get data from artifacts
    dir = get_data_path() / "interim"
    df_edges = pd.read_parquet(Path(dir) / "edge_list.parquet")

    df_edges = set_types_edge_list(df_edges)

    df_edges_clean = clean_edges(df_edges, min_dur_secs=min_dur_secs)

    df_edges_per_vesseltype = group_edges_per_vesseltype(df_edges_clean)

    save_path = get_data_path() / "interim"

    df_edges_per_vesseltype.to_parquet(save_path / "edge_list_aggregated.parquet")
    pd.read_parquet(save_path / "edge_list_aggregated.parquet")
