
#%%
from networkx.classes.function import density
from networkx.classes.graphviews import generic_graph_view
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import wandb
from pathlib import Path
from vessel_proj.data import read_edge_list, get_wandb_root_path, get_project_name

import argh

# %%


def df_edges_to_centr(df_edges):   
    #%% Check for weird links
    ind_no_dur = df_edges["duration_days"]==0
    print(f"found {ind_no_dur.sum() } links with zero duration" )

    ind_same_ports = df_edges["start_port"] == df_edges["end_port"]
    print(f"found {ind_same_ports.sum() } links with zero duration") 

    # drop them
    ind_to_drop = ind_no_dur | ind_same_ports
    print(f"dropping  {ind_to_drop.sum() } links ") 
    df_edges = df_edges[~ind_to_drop]

    #%% simple description numerical
    df_edges.describe()
    #%% simple description categorical
    df_edges.describe(include={"category"})

    #%% group trips
    # count number of connections between each pairs of ports, avg duration, number of distinct vessels and types of vessels

    count_unique =  lambda x: np.unique(x).shape[0]
    df_edges_grouped = df_edges.groupby(["start_port", "end_port"], observed=True).agg( duration_avg_days = ("duration_days", np.mean), trips_count =  ("uid", count_unique), vesseltype_count = ("vesseltype", count_unique))

    df_edges_grouped["log_trips_count"] = np.log(df_edges_grouped["trips_count"])

    df_edges_grouped.reset_index(inplace=True)
    # df_edges_grouped.drop(columns="index", inplace=True)

    df_edges_grouped.plot.scatter("vesseltype_count", "trips_count")
    df_edges_grouped.hist(log=True, density=True)


    #%% Create graph 
    G_0 = nx.convert_matrix.from_pandas_edgelist(df_edges_grouped, 'start_port', 'end_port',  edge_attr=["trips_count", "log_trips_count", "vesseltype_count", "duration_avg_days"], create_using=nx.DiGraph())
    #%% Drop smallest component

    subgraphs_conn = [G_0.subgraph(c).copy() for c in nx.connected_components(G_0.to_undirected())]

    n_nodes_conn = [S.number_of_nodes() for S in subgraphs_conn]

    print(f"Nodes in conn comp {n_nodes_conn}")
    G = subgraphs_conn[0]

    #%% Compute centralities

    df_centr = pd.DataFrame.from_dict(nx.eigenvector_centrality_numpy(G), orient="index", columns = ['centr_eig_bin'])

    df_centr["centr_eig_w_trips"] = pd.DataFrame.from_dict(nx.eigenvector_centrality_numpy(G, weight="trips_count"), orient="index").iloc[:, 0]
    df_centr["centr_eig_w_log_trips"] = pd.DataFrame.from_dict(nx.eigenvector_centrality_numpy(G, weight="log_trips_count"), orient="index").iloc[:, 0]


    df_centr["page_rank_bin"] = pd.DataFrame.from_dict(nx.algorithms.link_analysis.pagerank_alg.pagerank(G), orient="index").iloc[:, 0]
    df_centr["page_rank_w_trips"] = pd.DataFrame.from_dict(nx.algorithms.link_analysis.pagerank_alg.pagerank(G, weight="trips_count"), orient="index").iloc[:, 0]
    df_centr["page_rank_w_log_trips"] = pd.DataFrame.from_dict(nx.algorithms.link_analysis.pagerank_alg.pagerank(G, weight="log_trips_count"), orient="index").iloc[:, 0]


    if False:
        df_centr.plot.scatter("centr_eig_bin", "centr_eig_w_trips")
        df_centr.plot.scatter("centr_eig_w_log_trips", "centr_eig_w_trips")
        df_centr.plot.scatter("page_rank_bin", "page_rank_w_trips")
        df_centr.plot.scatter("centr_eig_bin", "page_rank_bin")

    return df_centr

def centralities():
    """load list of voyages (edge_list), clean the graph, compute a set of centralities and log them as parquet"""
    with wandb.init(project=get_project_name(), name="create_centrality_data", dir=get_wandb_root_path(), group="data_preprocessing", reinit=True) as run:

        #%% get data from artifacts
        art_edge = run.use_artifact(f"{get_project_name()}/edge_list:latest")
        dir = art_edge.download(root=get_wandb_root_path())
        df_edges = read_edge_list(Path(dir) / 'voyage_links.csv')
        df_centr = df_edges_to_centr(df_edges)
        art_centr = wandb.Artifact("centralities-ports", type="dataset", description="df with different centrality measures for both binary and weightsd voyages graph")
        with art_centr.new_file('centralities-ports.parquet', mode='wb') as file:
            df_centr.to_parquet(file)

        run.log_artifact(art_centr)

argh.dispatch_command(centralities)
