#%%
from prefect import flow, task
import pandas as pd
from pathlib import Path
import numpy as np
from vessel_proj.preprocess_data import get_data_path, set_types_edge_list

import logging
from prefect import task
from vessel_proj.preprocess_data import get_data_path, shaper_slow

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)


#%%
@task
def clean_visits(df, min_dur_sec=300, types_to_drop = ["fishing", "tug tow"]):
        
    inds = df["duration_seconds"] < min_dur_sec
    logger.info(f"Dropping {inds.sum()} rows for too low duration")
    df.drop(df[inds].index, inplace=True)

    
    inds = df["vessel_category"].apply(lambda x: x in types_to_drop)
    logger.info(f"Dropping {inds.sum()} rows because vessel type in {types_to_drop}")
    df.drop(df[inds].index, inplace=True)


    # store 
    file_path = get_data_path() / "interim" / 'visits-augmented-cleaned.parquet'
    df.to_parquet(file_path)

    return df


@task
def create_and_log_edge_list_from_visits(df_visits, nrows=None):


    output_file = get_data_path() / "interim" /'edge_list.parquet'

    if nrows is not None:
        df_visits = df_visits[:nrows]            
        output_file = Path(output_file)
        no_suff_name = output_file.name.replace(output_file.suffix, '')
        output_file = Path(output_file.parent) / ( f"{no_suff_name}_from_{nrows}_visits{output_file.suffix}"  )


    df_edges = shaper_slow(df_visits, output_file=output_file)

    return df_edges
  
@task
def load_visits():
    dir = get_data_path() / "raw" 
    df_visits = pd.read_csv(dir / 'visits-augmented.csv')
    df_visits.to_parquet(dir / 'visits-augmented.parquet')

    return df_visits




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


@task
def aggregated_graph_from_edge_list(df_edges, min_dur_secs=300):
    """
    from list of voyages (edge_list), clean the graph, compute a set of centralities and log them as parquet
    """

    df_edges = set_types_edge_list(df_edges)

    df_edges_clean = clean_edges(df_edges, min_dur_secs=min_dur_secs)

    df_edges_per_vesseltype = group_edges_per_vesseltype(df_edges_clean)

    save_path = get_data_path() / "interim"

    df_edges_per_vesseltype.to_parquet(save_path / "edge_list_aggregated.parquet")
    pd.read_parquet(save_path / "edge_list_aggregated.parquet")

#%%
@flow
def main():

    df_visits = load_visits.fn()

    df_visits_cleaned = clean_visits.fn(df_visits)

    df_edges = create_and_log_edge_list_from_visits.fn(df_visits_cleaned)

    aggregated_graph_from_edge_list.fn(df_edges)
    
if __name__ == '__main__':
    main()
# %%
