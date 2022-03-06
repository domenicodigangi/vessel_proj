#%%
import numpy as np
import pandas as pd
import networkx as nx
from prefect import task
import logging
from vessel_proj.preprocess_data import get_data_path

logger = logging.getLogger("root")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

from prefect import task

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


def group_links_all_categories(dfg_edges_per_cat):

    dfg_edges_per_cat = dfg_edges_per_cat.reset_index(level=2)

    df_edges = dfg_edges_per_cat.groupby(by = ["start_port", "end_port"]).agg(
        trips_count = ("trips_count", "sum"),
        vesseltype_count = ("vessel_category", "count"),
    )

    df_edges["log_trips_count"] = np.log(df_edges["trips_count"])

    df_edges = df_edges.reset_index()


    return df_edges


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

    
def get_centralities(G: nx.DiGraph) -> pd.DataFrame:
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
def main(load_path, save_path):
    """
    load edge_list  select largest connected component 
    """


    dfg_edges_per_cat = pd.read_parquet(load_path / "edge_list_aggregated.parquet")

    df_edges = group_links_all_categories(dfg_edges_per_cat)

    G = edges_to_nx_one_conn_comp(df_edges)

    df_centr = get_centralities(G)


    df_centr.to_parquet(save_path / "centralities-ports.parquet")

    return df_centr

if __name__ == "__main__":
    main()

