#%%
import pandas as pd
import numpy as np
import networkx as nx
import wandb
from pathlib import Path
from vessel_proj.preprocess_data import set_types_edge_list, get_wandb_root_path, get_project_name
from prefect import task
from . import get_data_path

import argh
import logging

logger = logging.getLogger("root")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

from prefect import task, flow

# %%


def download_graph_data_zenodo() -> pd.DataFrame:
    raise NotImplementedError

def get_graph_data() -> pd.DataFrame:
    graph_data_file = get_data_path() / "interim" / "edge_list_aggregated.parquet"

    try:
        df = pd.read_parquet(graph_data_file)
    except:
        df = download_graph_data_zenodo()

    return df


def clean_group_edges(df_edges, min_dur_secs=300):
    df_edges = set_types_edge_list(df_edges)

    df_edges_clean = clean_edges(df_edges, min_dur_secs=min_dur_secs)

    df_edges_per_vesseltype = group_edges_per_vesseltype(df_edges_clean)

    dfg_edges = aggregate_vesseltypes(df_edges_per_vesseltype)

    return dfg_edges


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

def aggregate_vesseltypes(df_edges_per_vesseltype: pd.DataFrame) -> pd.DataFrame:
    #%% group trips
    # count number of connections between each pairs of ports, avg duration, number of distinct vessels and types of vessels

    df_edges_grouped = df_edges_per_vesseltype.groupby(["start_port", "end_port"], observed=True).agg(
        duration_avg_days=("duration_avg_days", np.mean),
        trips_count=("trips_count", np.sum),
        vesseltype_count=("trips_count", "count")
    )

    df_edges_grouped["log_trips_count"] = np.log(df_edges_grouped["trips_count"])

    df_edges_grouped.reset_index(inplace=True)
    # df_edges_grouped.drop(columns="index", inplace=True)

    df_edges_grouped.plot.scatter("vesseltype_count", "trips_count")
    df_edges_grouped.hist(log=True, density=True)

    return df_edges_grouped


def edges_to_nx_one_conn_comp(df_edges_grouped: pd.DataFrame):
    #%% Create graph
    G_0 = nx.convert_matrix.from_pandas_edgelist(
        df_edges_grouped,
        "start_port",
        "end_port",
        edge_attr=[
            "trips_count",
            "log_trips_count",
            "vesseltype_count",
            "duration_avg_days",
        ],
        create_using=nx.DiGraph(),
    )

    #%% keep only largest component

    subgraphs_conn = [
        G_0.subgraph(c).copy() for c in nx.connected_components(G_0.to_undirected())
    ]

    n_nodes_conn = [S.number_of_nodes() for S in subgraphs_conn]

    logger.info(f"Nodes in conn comp {n_nodes_conn}")
    G = subgraphs_conn[0]

    return G


def get_centralities(G) -> pd.DataFrame:
    #%% Compute centralities
    df_centr = pd.DataFrame.from_dict(
        nx.eigenvector_centrality_numpy(G), orient="index", columns=["centr_eig_bin"]
    )

    df_centr["centr_eig_w_trips"] = pd.DataFrame.from_dict(
        nx.eigenvector_centrality_numpy(G, weight="trips_count"), orient="index"
    ).iloc[:, 0]
    df_centr["centr_eig_w_log_trips"] = pd.DataFrame.from_dict(
        nx.eigenvector_centrality_numpy(G, weight="log_trips_count"), orient="index"
    ).iloc[:, 0]

    df_centr["page_rank_bin"] = pd.DataFrame.from_dict(
        nx.algorithms.link_analysis.pagerank_alg.pagerank(G), orient="index"
    ).iloc[:, 0]
    df_centr["page_rank_w_trips"] = pd.DataFrame.from_dict(
        nx.algorithms.link_analysis.pagerank_alg.pagerank(G, weight="trips_count"),
        orient="index",
    ).iloc[:, 0]
    df_centr["page_rank_w_log_trips"] = pd.DataFrame.from_dict(
        nx.algorithms.link_analysis.pagerank_alg.pagerank(G, weight="log_trips_count"),
        orient="index",
    ).iloc[:, 0]

    df_centr["closeness_bin"] = pd.DataFrame.from_dict(
        nx.algorithms.centrality.closeness_centrality(G), orient="index"
    ).iloc[:, 0]

    df_centr["betweenness_bin"] = pd.DataFrame.from_dict(
        nx.algorithms.centrality.betweenness_centrality(G), orient="index"
    ).iloc[:, 0]

    if False:
        df_centr.plot.scatter("centr_eig_bin", "centr_eig_w_trips")
        df_centr.plot.scatter("centr_eig_w_log_trips", "centr_eig_w_trips")
        df_centr.plot.scatter("page_rank_bin", "page_rank_w_trips")
        df_centr.plot.scatter("centr_eig_bin", "page_rank_bin")

    return df_centr

#%%

@task
def main(min_dur_secs=300):
    """load list of voyages (edge_list), clean the graph, compute a set of centralities and log them as parquet"""

        #%% get data from artifacts
        art_edge = run.use_artifact(f"{get_project_name()}/edge_list:latest")
        dir = art_edge.download(root=get_wandb_root_path())
        df_edges = pd.read_parquet(Path(dir) / "edge_list.parquet")

        dfg_edges = clean_group_edges(df_edges, min_dur_secs=min_dur_secs)

        G = edges_to_nx_one_conn_comp(dfg_edges)

        df_centr = get_centralities(G)

        art_centr = wandb.Artifact(
            "centralities-ports",
            type="dataset",
            description="df with different centrality measures for both binary and weightsd voyages graph",
        )
        with art_centr.new_file("centralities-ports.parquet", mode="wb") as file:
            df_centr.to_parquet(file)

        run.log_artifact(art_centr)


if __name__ == "__main__":
    argh.dispatch_command(main)

