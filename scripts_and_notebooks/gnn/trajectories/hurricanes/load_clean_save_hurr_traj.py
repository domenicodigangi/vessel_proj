#%%
from enum import unique
from typing import List, Tuple
import pandas as pd
from geopy import distance
from vessel_proj.ds_utils import get_data_path
import numpy as np
import networkx as nx
from pathlib import Path
import datetime
import csv


#%%

loadfile = get_data_path() / "raw" / "hurricanes" / "hurdat2-1851-2021-100522.txt"
savefoldpath = Path(
    "/home/digan/cnr/vessel_proj/data/interim/hurricane_traj_preprocessed"
)
savefoldpath.mkdir(exist_ok=True)

df_all_traj = pd.read_csv(loadfile)

df_all_traj["g"] = df_all_traj.iloc[:, 3].isna().cumsum()

df_all_list = [df for g, df in df_all_traj.groupby(by="g")]


with open(loadfile, newline="") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=",")
    traj = []
    traj_list = []
    headers_list = []
    for row in spamreader:
        if row[0].startswith("A"):
            headers_list.append(row)
            traj_list.append(traj)
            traj = []
        else:
            traj.append(row)

    traj_list.append(traj)


def hurr_category_from_max_wind(max_wind_knt: int) -> int:
    if max_wind_knt < 64:
        return 0
    elif max_wind_knt < 83:
        return 1
    elif max_wind_knt < 96:
        return 2
    elif max_wind_knt < 112:
        return 3
    elif max_wind_knt < 136:
        return 4
    else:
        return 5


df_all = pd.DataFrame()
for head, traj in zip(headers_list, traj_list[1:]):
    df = pd.DataFrame(traj)
    df["id"] = head[0]
    df["name"] = head[1]
    df["number_rows"] = head[2]
    df["unknown"] = head[3]
    df = df.rename(
        columns={
            0: "date",
            1: "time",
            2: "record_type",
            3: "status",
            4: "latitude_str",
            5: "longitude_str",
            6: "max_wind",
            7: "min_pressure",
        }
    )
    df["max_wind"] = df["max_wind"].astype(int)
    df["cat"] = hurr_category_from_max_wind(df["max_wind"].max())

    df_all = pd.concat((df_all, df))


df_all = df_all[
    [
        "id",
        "date",
        "time",
        "latitude_str",
        "longitude_str",
        "cat",
    ]
]

df_all["datetime"] = pd.to_datetime(
    pd.to_datetime(df_all["date"]).astype(str) + " " + df_all["time"]
)
df_all = df_all.reset_index().drop(columns="index")


# %%
inds_north = df_all["latitude_str"].str.contains("N")
inds_south = df_all["latitude_str"].str.contains("S")
df_all["latitude"] = 0.0
if inds_north.any():
    df_all.loc[inds_north, "latitude"] = (
        df_all["latitude_str"][inds_north].str.replace("N", "").astype(float)
    )
if inds_south.any():
    df_all.loc[inds_south, "latitude"] = (
        -df_all["latitude_str"][inds_south].str.replace("S", "").astype(float)
    )
inds_west = df_all["longitude_str"].str.contains("W")
inds_east = df_all["longitude_str"].str.contains("E")
df_all["longitude"] = 0.0
if inds_west.any():
    df_all.loc[inds_west, "longitude"] = (
        df_all["longitude_str"][inds_west].str.replace("W", "").astype(float)
    )
if inds_east.any():
    df_all.loc[inds_east, "longitude"] = (
        -df_all["longitude_str"][inds_east].str.replace("E", "").astype(float)
    )


df_all["lat_long"] = df_all[["latitude", "longitude"]].apply(tuple, axis=1)


df_all = df_all.drop(df_all[df_all["latitude"] > 90].index)
df_all = df_all.drop(df_all[df_all["latitude"] < -90].index)


for g, df in df_all.groupby(by="id"):
    if df.shape[0] > 1:
        df["lat_long_prev"] = df["lat_long"].shift(1)
        df["one_step_distance_meters"] = (
            df[["lat_long", "lat_long_prev"]]
            .dropna()
            .apply(
                lambda x: distance.distance(x["lat_long"], x["lat_long_prev"]).m, axis=1
            )
        )

        df["delta_datetime"] = df["datetime"] - df["datetime"].shift(1)
        df["delta_date_seconds"] = df["delta_datetime"].dt.seconds
        df["speed_m_s"] = df["one_step_distance_meters"] / df["delta_date_seconds"]

        df["datetime"] = df["datetime"].astype("datetime64[ms]")
        df.to_parquet(savefoldpath / f"{g}.parquet")

# %%
