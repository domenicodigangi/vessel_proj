import pandas as pd
from pathlib import Path
import numpy as np
from vessel_proj.ds_utils import get_data_path

import logging

from vessel_proj.preprocess_data import set_types_edge_list, shaper

logger = logging.getLogger("root")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)


def clean_visits(df, min_dur_sec=300, types_to_drop=["fishing", "tug tow"]):

    inds = df["duration_seconds"] < min_dur_sec
    logger.info(f"Dropping {inds.sum()} rows for too low duration")
    df.drop(df[inds].index, inplace=True)

    inds = df["vessel_category"].apply(lambda x: x in types_to_drop)
    logger.info(f"Dropping {inds.sum()} rows because vessel type in {types_to_drop}")
    df.drop(df[inds].index, inplace=True)

    # store
    file_path = get_data_path() / "interim" / "visits-augmented-cleaned.parquet"
    df.to_parquet(file_path)

    return df


def create_and_save_edge_list_from_visits(df_visits, nrows=None):

    output_file = get_data_path() / "raw" / "edge_list.parquet"

    if nrows is not None:
        df_visits = df_visits[:nrows]
        output_file = Path(output_file)
        no_suff_name = output_file.name.replace(output_file.suffix, "")
        output_file = Path(output_file.parent) / (
            f"{no_suff_name}_from_{nrows}_visits{output_file.suffix}"
        )

    df_edges = shaper(df_visits, output_file=output_file)

    # df_edges_1 = shaper_slow(df_visits, output_file=output_file)
    # pd.testing.assert_frame_equal(df_edges.reset_index().drop(columns="index"), df_edges_1.reset_index().drop(columns="index"))

    return df_edges


def load_visits():
    dir = get_data_path() / "raw" / "vessel_visits"
    try:
        df_visits = pd.read_parquet(dir / "visits-augmented.parquet")
    except FileNotFoundError:
        df_visits = pd.read_csv(dir / "visits-augmented.csv")
        df_visits.to_parquet(dir / "visits-augmented.parquet")

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


def group_edges_per_vessel_category(df_edges: pd.DataFrame) -> pd.DataFrame:
    # count number of connections between each pairs of ports, avg duration, number of distinct vessels and types of vessels
    logger.info("group trips ")

    def count_unique(x):
        return np.unique(x).shape[0]

    df_edges_per_vessel_category = df_edges.groupby(
        ["start_port", "end_port", "vessel_category"], observed=True
    )[["uid"]].agg(
        # duration_avg_days=("duration_days", np.mean),
        trips_count=("uid", count_unique),
    )

    return df_edges_per_vessel_category


def aggregated_graph_from_edge_list(df_edges, min_dur_secs=3600):
    """
    from list of voyages (edge_list), clean the graph, compute a set of centralities and log them as parquet
    """

    df_edges = set_types_edge_list(df_edges)

    df_edges = clean_edges(df_edges, min_dur_secs=min_dur_secs)

    df_edges_per_vessel_category = group_edges_per_vessel_category(df_edges)

    save_path = get_data_path() / "interim"

    df_edges_per_vessel_category.to_parquet(save_path / "edge_list_aggregated.parquet")
    # pd.read_parquet(save_path / "edge_list_aggregated.parquet")

    return df_edges_per_vessel_category


def main():

    df_visits = load_visits()

    df_visits_cleaned = clean_visits(df_visits)

    df_edges = create_and_save_edge_list_from_visits(df_visits_cleaned)

    df_edges_per_vessel_category = aggregated_graph_from_edge_list(
        df_edges, min_dur_secs=3600
    )


if __name__ == "__main__":
    main()
