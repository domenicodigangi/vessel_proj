#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import logging
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
from vessel_proj.preprocess_data import get_latest_port_data_task
import seaborn as sns

sns.set_theme(style="darkgrid")

from vessel_proj.task.utils_classification_task_pipeline_ports_centr import (
    encode_features,
    drop_missing_cols,
)

#%% get data from artifacts
data = get_latest_port_data_task.fn("cargo")

df = data["features"]
df_enc = encode_features.fn(
    data,
    feat_names_non_cat=["TIDE_RANGE", "LATITUDE", "LONGITUDE"],
    cols_to_drop=["PORT_NAME", "REGION_NO", "PUB"],
)["features"]

df.shape
#%% Missing
fract_miss = df.isnull().mean().sort_values(ascending=False)

df.columns[df.dtypes == np.float64]
df["TIDE_RANGE"].hist()

fig, ax = plt.subplots(figsize=(10, 12))
g = sns.barplot(y=fract_miss.index, x=fract_miss.values, ax=ax)
g.tick_params(axis="x", rotation=90)
g.set_xlabel("Fraction of Missing Values")
#%%
df_enc[["CARGODEPTH"]].merge(
    df[["CARGODEPTH"]], on="INDEX_NO"
).drop_duplicates().sort_values(by="CARGODEPTH_x")

# %%
df_enc["COMM_AIR"].dropna().value_counts()
