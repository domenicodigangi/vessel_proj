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


#%%

loadfile = get_data_path() / "raw" / "hurricanes" / "hurdat2-1851-2021-100522.txt"
savefoldpath = Path(
    "/home/digan/cnr/vessel_proj/data/interim/hurricane_traj_preprocessed"
)
savefoldpath.mkdir(exist_ok=True)

df_all_traj = pd.read_csv(loadfile)

df_all_traj["g"] = df_all_traj.iloc[:, 3].isna().cumsum()

df_all_list = [df for g, df in df_all_traj.groupby(by="g")]

len(df_all_list)
df_all_traj.head(20)

df = df_all_list[3]

df
df.shape

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

df_all = pd.DataFrame()
for head, traj in zip(headers_list, traj_list[1:]):
    df = pd.DataFrame(traj)
    df["id"] = head[0]
    df["name"] = head[1]
    df["number_rows"] = head[2]
    df["unknown"] = head[3]
    df_all = pd.concat((df_all, df))


df_final = df_all.rename(
    columns={
        0: "date",
        1: "time",
        2: "record_type",
        3: "status",
        4: "latitude",
        5: "longitude",
        6: "max_wind",
        7: "min_pressure",
    }
)[
    [
        "id",
        "date",
        "time",
        "record_type",
        "status",
        "latitude",
        "longitude",
        "max_wind",
        "min_pressure",
    ]
]

# TODO add classification
# TODO compute distances
# TODO compute speed
