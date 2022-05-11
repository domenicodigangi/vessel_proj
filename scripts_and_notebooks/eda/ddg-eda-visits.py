# %%
import plotly.express as px
from vessel_proj.preprocess_data import get_latest_port_data
import pandas as pd
import numpy as np
from pathlib import Path
from vessel_proj.ds_utils import get_project_name, get_wandb_root_path
import wandb
import seaborn as sns
sns.set_theme(style="darkgrid")

# %% get data from artifacts
proj_name = get_project_name()

api = wandb.Api()

dir = api.artifact(
    f"{proj_name}/all_raw_data:latest").checkout(root=get_wandb_root_path()/"all_raw_data")

df_visits = pd.read_parquet(Path(dir) / 'visits-augmented.parquet')

data = get_latest_port_data()
df_feat = data["features"]
df_centr = data["centralities"]

# %% Scatter of weighted vs binary centr
df = df_visits

df.drop(df[df["duration_seconds"] == 0].index, inplace=True)


(df["duration_seconds"] < 3600).mean()
(df["duration_seconds"] < 300).mean()
df["duration_min"] = df["duration_seconds"]/60

np.log10(df["duration_seconds"]).hist()


dfg = df.groupby(by="port").agg(
    count=("port_name", "count"),
    port_name=("port_name", "first"),
    dur_mean=("duration_min", "mean"),
    dur_median=("duration_min", lambda x: x.quantile(0.5)),
    dur_05=("duration_min", lambda x: x.quantile(0.5)),
    dur_95=("duration_min", lambda x: x.quantile(0.95))
)

dfg["log10_count"] = np.log10(dfg["count"])

dfg = dfg.merge(df_centr, left_on="port", right_index=True)

dfg.sort_values(by="count", inplace=True, ascending=False)

sns.jointplot(data=dfg, x="log10_count", y="dur_median")

fig = px.scatter(dfg,  x="log10_count", y="dur_median", hover_data=[
                 "count", "port_name"], size="page_rank_w_log_trips",  width=1000, height=800)
fig.show()


# %%
(np.log10(df["count"][df["duration_seconds"] < 300])).hist()
