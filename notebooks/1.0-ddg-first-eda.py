import pandas as pd
import tdqm

from src.utils.data import get_raw_data_path, get_project_root


get_project_root()

df_vis = pd.read_csv(get_raw_data_path() /'visits-augmentend.csv')

df_ports = pd.read_csv(get_raw_data_path() / "ports-1000.csv")


import os


os.listdir(get_raw_data_path())

# To Do:
# - la funzione di emanuele passa da visite a edges per una vecchia versione della tab sulle visite. Aggiornare alla nuova versione dei dati (unica che io ho visto)
# - usare uid invece di mmsi per fare la group by
# - portarsi dietro i dati su region, etc..


input_file = get_raw_data_path() /'visits-augmentend.csv'

# segue codice copiato da Emanuele 
def shaper_slow(input_file, output_file):
    
    df_compact = pd.read_csv(input_file)
    df_compact['start'] = pd.to_datetime(df_compact['start']) 
    df_compact['end'] = pd.to_datetime(df_compact['end']) 
    
    
    l = []
    groups = df_compact.groupby('mmsi')

    for name, dfg in tqdm(groups, desc="Computing edge lists", position=0, leave=True):
        for i in range(0, len(dfg)):

            if i+1 < len(dfg):            
                l.append({'start_port':dfg.iloc[i].port,
                 'start_date':dfg.iloc[i].end,
                 'end_port':dfg.iloc[i+1].port,
                 'end_date':dfg.iloc[i+1].start,
                 'duration':dfg.iloc[i+1].start - dfg.iloc[i].end,
                 'mmsi':dfg.iloc[i].mmsi,
                 ##'cargo':dfg.iloc[i].cargo,
                 'vesseltype':dfg.iloc[i].vesseltype,
                })


    df_edges = pd.DataFrame(data = l)
    df_edges.to_csv(output_file)