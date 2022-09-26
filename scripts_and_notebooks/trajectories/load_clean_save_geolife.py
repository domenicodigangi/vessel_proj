import pandas as pd
from geopy import distance

from vessel_proj.ds_utils import get_data_path

load_path = get_data_path() / "raw" / "Geolife Trajectories 1.3" / "Data"


df = pd.read_csv(
    "/home/digan/cnr/vessel_proj/data/raw/Geolife Trajectories 1.3/Data/179/Trajectory/20080823012829.plt",
    skiprows=6,
    header=None,
    names=[
        "latitude",
        "longitude",
        "na",
        "altitude",
        "date_float",
        "date_str",
        "time_str",
    ],
)

df = df.drop(columns=["na", "date_str", "time_str"])

df = df.drop_duplicates()

n_min_points_per_traj = 100
n_5_min_intervals_in_one_day = 288
n_seconds_in_one_day = 60 * 60 * 24
max_delta = 1 / n_5_min_intervals_in_one_day
df["traj_group"] = (df["date_float"].diff() > max_delta).cumsum()
df["traj_group"].iloc[0] = df["traj_group"].iloc[1]

df["lat_long"] = df[["latitude", "longitude"]].apply(tuple, axis=1)

list(filter(lambda x: x[1].shape[0] > n_min_points_per_traj, df.groupby("traj_group")))


df["lat_long_prev"] = df["lat_long"].shift(-1)

df["distance_meters"] = (
    df[["lat_long", "lat_long_prev"]]
    .dropna()
    .apply(lambda x: distance.distance(x["lat_long"], x["lat_long_prev"]).m, axis=1)
)

df["delta_date_float"] = df["date_float"].shift(-1) - df["date_float"]
df["delta_date_seconds"] = df["delta_date_float"] * n_seconds_in_one_day
df["speed_m_s"] = df["distance_meters"] / df["delta_date_seconds"]

for i, row in df.iterrows():
    row["lat_long"]
    df["lat_long"].apply(lambda x: distance.distance(x, row["lat_long"]).m)

# TODO define criterium to link two trajectory points

# TODO Find doable strategy to get graph from trajectory
# TODO add category
# TODO save in parquet


# TODO make function
# TODO run function on all files
