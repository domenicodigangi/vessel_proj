from pathlib import Path
import logging
from typing import Optional
import pandas as pd
from tqdm import tqdm
import logging
import wandb
from pathlib import Path
from types import SimpleNamespace
from prefect import task
from dotenv import load_dotenv
import os
from vessel_proj.ds_utils import get_data_path


logger = logging.getLogger()
logger.setLevel(logging.INFO)


load_dotenv()


def get_one_file_from_artifact(name, run=None, type=None):

    if run is not None:
        artifact = run.use_artifact(name, type=type)
    else:
        api = wandb.Api()
        fullname = f"{get_project_name()}/{name}"
        artifact = api.artifact(fullname, type=type)

    d = {"artifact": artifact}

    try:
        artifact_dir = artifact.download(
            root=get_wandb_root_path() / artifact._default_root()
        )
        d["artifact_dir"] = artifact_dir
    except:
        pass

    try:
        manifest = artifact._load_manifest()
        filepath = artifact_dir / list(manifest.entries)[0]
        d["filepath"] = filepath
    except:
        pass

    out = SimpleNamespace(**d)

    return out


def shaper_slow(df_visits, output_file=None):
    """
    Convert df of time stamped vessel visits to ports to links, using a for loop over the groups of visits for each vessel

    """

    df_visits["start"] = pd.to_datetime(df_visits["start"])
    df_visits["end"] = pd.to_datetime(df_visits["end"])

    l = []
    groups = df_visits.groupby("uid")

    for name, dfg in tqdm(groups, desc="Computing edge lists", position=0, leave=True):
        for i in range(0, len(dfg)):

            if i + 1 < len(dfg):
                l.append(
                    {
                        "start_port": dfg.iloc[i].port,
                        "start_region": dfg.iloc[i].region,
                        "start_port_name": dfg.iloc[i].port_name,
                        "start_date": dfg.iloc[i].end,
                        "end_port": dfg.iloc[i + 1].port,
                        "end_region": dfg.iloc[i + 1].region,
                        "end_port_name": dfg.iloc[i + 1].port_name,
                        "end_date": dfg.iloc[i + 1].start,
                        "duration": dfg.iloc[i + 1].start - dfg.iloc[i].end,
                        "mmsi": dfg.iloc[i].mmsi,
                        "uid": dfg.iloc[i].uid,
                        # 'cargo':dfg.iloc[i].cargo,
                        "vesseltype": dfg.iloc[i].vesseltype,
                        "vessel_category": dfg.iloc[i].vessel_category,
                    }
                )

    df_edges = pd.DataFrame(data=l)

    df_edges["duration_seconds"] = df_edges["duration"].dt.total_seconds()
    df_edges.drop(columns=["duration"], inplace=True)

    if output_file is not None:
        if ".csv" in str(output_file):
            df_edges.to_csv(output_file)
        elif ".parquet" in str(output_file):
            df_edges.to_parquet(output_file)

    return df_edges


def shaper(df_visits, output_file=None):
    """
    Convert df of time stamped vessel visits to ports to links, using a for loop over the groups of visits for each vessel

    """

    df_visits["start"] = pd.to_datetime(df_visits["start"])
    df_visits["end"] = pd.to_datetime(df_visits["end"])
    groups = df_visits.groupby("uid")

    l = []
    for name, dfg in tqdm(groups, desc="Computing edge lists", position=0, leave=True):

        # dfg = groups.get_group(list(groups.groups)[100])
        dfg["start_port"] = dfg["port"]
        dfg["start_region"] = dfg["region"]
        dfg["start_port_name"] = dfg["port_name"]
        dfg["start_date"] = dfg["end"]
        dfg["end_port"] = dfg["port"].shift(-1, fill_value=0)
        dfg["end_region"] = dfg["region"].shift(-1)
        dfg["end_port_name"] = dfg["port_name"].shift(-1)
        dfg["end_date"] = dfg["start"].shift(-1)
        dfg["duration"] = dfg["end_date"] - dfg["start_date"]

        df_app = dfg.reset_index()[
            [
                "start_port",
                "start_region",
                "start_port_name",
                "start_date",
                "end_port",
                "end_region",
                "end_port_name",
                "end_date",
                "duration",
                "mmsi",
                "uid",
                "vesseltype",
                "vessel_category",
            ]
        ]

        l.append(df_app[:-1])

    df_edges = pd.concat(l)

    df_edges["duration_seconds"] = df_edges["duration"].dt.total_seconds()
    df_edges.drop(columns=["duration"], inplace=True)

    if output_file is not None:
        if ".csv" in str(output_file):
            df_edges.to_csv(output_file)
        elif ".parquet" in str(output_file):
            df_edges.to_parquet(output_file)

    return df_edges


def set_types_edge_list(df_edges):
    df_edges["duration_days"] = df_edges["duration_seconds"] / (60 * 60 * 24)
    df_edges["vesseltype"] = df_edges["vesseltype"].astype("category")
    df_edges["vessel_category"] = df_edges["vessel_category"].astype("category")
    df_edges["start_port"] = df_edges["start_port"].astype("category")
    df_edges["end_port"] = df_edges["end_port"].astype(df_edges["start_port"].dtype)
    df_edges["start_port_name"] = df_edges["start_port_name"].astype("category")
    df_edges["end_port_name"] = df_edges["end_port_name"].astype(
        df_edges["start_port_name"].dtype
    )
    df_edges["start_region"] = df_edges["start_region"].astype("category")
    df_edges["end_region"] = df_edges["end_region"].astype(df_edges["end_region"].dtype)
    df_edges["mmsi"] = df_edges["mmsi"].astype("category")
    df_edges["uid"] = df_edges["uid"].astype("category")
    return df_edges


@task
def get_latest_port_data_task(vessel_category, load_path=None):

    data = {
        "centralities": get_latest_port_centralities(vessel_category, load_path),
        "features": get_latest_port_features(load_path),
    }

    return data


def get_latest_port_centralities(
    vessel_category, load_path: Optional[Path] = None
) -> pd.DataFrame:
    if load_path is None:
        load_path = get_data_path() / "processed"

    centralities = pd.read_parquet(
        load_path / f"centralities-ports-{vessel_category}.parquet"
    )

    return centralities


def get_latest_port_features(load_path: Optional[Path] = None) -> pd.DataFrame:

    if load_path is None:
        load_path = get_data_path() / "processed"

    features = pd.read_parquet(load_path / f"ports_features.parquet")

    return features
