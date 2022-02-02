#%%
import fractions
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import logging
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import sklearn

from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.exceptions import ConvergenceWarning
import warnings

from torch import frac_
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from vessel_proj.data import get_one_file_from_artifact, get_project_name, get_wandb_root_path
import wandb
from sklearn.inspection import permutation_importance

from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from vessel_proj.data import get_latest_port_data
from sklearn.cluster import KMeans
import seaborn as sns
sns.set_theme(style="darkgrid")

from vessel_proj.task.classification_task_pipeline_ports_centr import encode_features, drop_missing_cols
#%% get data from artifacts

data = get_latest_port_data()

df = encode_features.fn(data)["features"]

#%% Missing
fract_miss = df.isnull().mean().sort_values(ascending=False)


fig, ax = plt.subplots(figsize=(10,12))
g = sns.barplot(y=fract_miss.index, x=fract_miss.values, ax=ax)
g.tick_params(axis='x', rotation=90)
g.set_xlabel("Fraction of Missing Values")
#%%

df = drop_missing_cols.fn({"features": df})["features"]


df.info()

# %%
