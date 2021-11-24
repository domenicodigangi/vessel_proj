import pandas as pd
import networkx as nx

import geopandas as gpd
from src.data import get_data_path

#%% load data
visist_file = get_data_path() / "raw" /'visits-augmentend.csv'
links_file = get_data_path() / "raw" /'voyage_links_from_100000_visits.csv'

df_links = pd.read_csv(links_file)
df_ports = gpd.read_file(get_data_path() / "raw" / "wpi" / 'WPI.shp' )

df_ports["INDEX_NO"] = df_ports["INDEX_NO"].astype('int')

df_ports = df_ports.set_index("INDEX_NO")




#%% Create graph and compute centralities

G = nx.from_pandas_edgelist(df_links, 'start_port', 'end_port')
df_centr = pd.DataFrame.from_dict(nx.eigenvector_centrality_numpy(G), orient="index", columns = ['centr_eig'])

df_centr['centr_deg_und'] = pd.DataFrame.from_dict(nx.degree_centrality(G), orient="index")



# %% 
df_reg = pd.concat((df_centr, df_ports), axis=1)