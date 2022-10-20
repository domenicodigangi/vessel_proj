#%%

import pandas as pd
from geopy import distance
from vessel_proj.ds_utils import get_data_path
from pathlib import Path
import numpy as np

#%%
loadfoldpath = get_data_path() / "raw" / "vessel_test_data"

savefoldpath = Path(
    "/home/digan/cnr/vessel_proj/data/interim/vessel_test_traj_preprocessed"
)
savefoldpath.mkdir(exist_ok=True)

df_anchored = pd.read_csv(loadfoldpath / "anchored.csv")
df_anchored["cat"] = "anchored"
df_moored = pd.read_csv(loadfoldpath / "moored.csv")
df_moored["cat"] = "moored"
df_underway = pd.read_csv(loadfoldpath / "underway.csv")
df_underway["cat"] = "underway"

df_all = pd.concat((df_anchored, df_moored, df_underway))

df_all["datetime"] = pd.to_datetime(df_all["timestamp"])
df_all["datetime"] = df_all["datetime"].astype("datetime64[ms]")

df_all = df_all.reset_index().drop(columns="index")
df_all["lat_long"] = df_all[["latitude", "longitude"]].apply(tuple, axis=1)

df_all = df_all.drop(df_all[df_all["latitude"] > 90].index)
df_all = df_all.drop(df_all[df_all["latitude"] < -90].index)

for g, df in df_all.groupby(by=["event_id", "cat"]):

    df["lat_long_prev"] = df["lat_long"].shift(1)

    df["one_step_distance_meters"] = (
        df[["lat_long", "lat_long_prev"]]
        .dropna()
        .apply(lambda x: distance.distance(x["lat_long"], x["lat_long_prev"]).m, axis=1)
    )

    df["delta_datetime"] = df["datetime"] - df["datetime"].shift(1)
    df["delta_date_seconds"] = df["delta_datetime"].dt.seconds
    df["speed_m_s"] = df["one_step_distance_meters"] / df["delta_date_seconds"]
    
    df = df.loc[~df["speed_m_s"].isin([np.inf, -np.inf])]

    df.to_parquet(savefoldpath / f"{g[0]}_{g[1]}.parquet")

# %%
