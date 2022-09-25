import pandas as pd
import geopandas
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
max_delta = 1 / n_5_min_intervals_in_one_day
df["traj_group"] = (df["date_float"].diff() > max_delta).cumsum()
df["traj_group"].iloc[0] = df["traj_group"].iloc[1]

list(filter(lambda x: x[1].shape[0] > n_min_points_per_traj, df.groupby("traj_group")))

# TODO compute speed,

gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.longitude, df.latitude)
)

gdf["geometry"].distance(gdf["geometry"].shift())

# TODO add category
# TODO save in parquet


# TODO make function
# TODO run function on all files
