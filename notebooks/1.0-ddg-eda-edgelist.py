#%%
from networkx.classes.function import density
from networkx.classes.graphviews import generic_graph_view
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from wandb.util import artifact_to_json

from vessel_proj.data import get_one_file_from_artifact, read_edge_list, get_wandb_root_path

get_wandb_root_path()


#%% get data from artifacts
name = "edge_list:latest"
art_edge = get_one_file_from_artifact('edge_list:latest')

df_edges = read_edge_list(art_edge.filepath)
df_edges.dtypes   

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

df_edges_grouped.reset_index(inplace=True)
# df_edges_grouped.drop(columns="index", inplace=True)

df_edges_grouped.plot.scatter("vesseltype_count", "trips_count")
df_edges_grouped.hist(log=True, density=True)


#%% Create graph 
G_0 = nx.convert_matrix.from_pandas_edgelist(df_edges_grouped, 'start_port', 'end_port',  edge_attr=["trips_count", "vesseltype_count", "duration_avg_days"], create_using=nx.DiGraph())
#%% Drop smallest component

subgraphs_conn = [G_0.subgraph(c).copy() for c in nx.connected_components(G_0.to_undirected())]

n_nodes_conn = [S.number_of_nodes() for S in subgraphs_conn]

print(f"Nodes in conn comp {n_nodes_conn}")
G = subgraphs_conn[0]

#%% Compute centralities

df_centr = pd.DataFrame.from_dict(nx.eigenvector_centrality_numpy(G), orient="index", columns = ['centr_eig_bin'])

df_centr["centr_eig_w"] = nx.eigenvector_centrality_numpy(G, weight="trips_count")
df_centr["page_rank_bin"] = nx.algorithms.link_analysis.pagerank_alg.pagerank(G)
df_centr["page_rank_w"] = nx.algorithms.link_analysis.pagerank_alg.pagerank(G, weight="trips_count")

df_centr.plot.scatter("centr_eig_bin", "centr_eig_w")

(df_centr["centr_eig_w"] == df_centr["page_rank_bin"]).mean()
(df_centr["page_rank_w"] == df_centr["page_rank_bin"]).mean()

# df_reg = pd.concat((df_centr, df_ports), axis=1)


#%%

if False:
    #log grouped data as artifact
    pass


# %% Drop some ports information

art_ports_info = get_one_file_from_artifact('ports_info_csv:latest')
df_ports = pd.read_csv(art_ports_info.filepath)


descr_num = df_ports.describe().transpose()
descr_obj = df_ports.describe(include = ["object"]).transpose()


col_to_drop = ["CHART", "geometry", "LAT_DEG", "LAT_MIN", "LONG_DEG", "LONG_MIN", "LAT_HEMI", "LONG_HEMI"]

col_single_val = df_ports.columns[df_ports.apply(lambda x: pd.unique(x).shape[0]) == 1].values.tolist()
print(col_single_val)

col_to_drop.extend(col_single_val)

df_ports = df_ports.drop(columns=col_to_drop)

df_ports["INDEX_NO"] = df_ports["INDEX_NO"].astype('int')
df_ports = df_ports.set_index("INDEX_NO")

#%% change types
# df_ports.astype({"PORT_NAME": df_edges["start_port_name"].dtype})

# %% EDA ports info
for col in df_ports.columns:
    if col not in ["PORT_NAME", "COUNTRY"]:
        plt.figure()
        df_ports[col].hist()
        plt.title(col)
        plt.show()
# %%
# TO DO:

# continuare a rimuovere colonne non interessanti
# cosa sono le COMM ?  e.g. COMM_RADIO
# add country dgp
# chiarire perchè le centralità pesate sono tutte uguali
# look at correlations
# decidere come trattare i missing

y_name = "centr_eig_bin"
df_reg = df_centr[y_name].reset_index().merge(df_ports.reset_index(), left_on="index", right_on="INDEX_NO")

df_reg = df_reg.astype({"COUNTRY": "category"})

df_reg.info()



        

cols_to_drop=["index", "PORT_NAME", "INDEX_NO", "Unnamed: 0", "REGION_NO"]
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import time

le = preprocessing.LabelEncoder()
X = df_reg.drop(columns=cols_to_drop + [y_name])
feature_names = [col for col in X.columns]
for col in X.columns:
    X[col] = le.fit_transform(X[col])

y = df_reg[y_name]#le.fit_transform(df_reg[y_name])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
forest = RandomForestRegressor(random_state=0)
forest.fit(X_train, y_train)

# %% Feature importance based on mean decrease in impurity

start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

# %%

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# %% Feature importance based on feature permutation
from sklearn.inspection import permutation_importance
import pickle

start_time = time.time()
result = permutation_importance(
    forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=20
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)

# %%
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()