#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import logging
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
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from vessel_proj.data import get_one_file_from_artifact, get_project_name, get_wandb_root_path
import wandb
from sklearn.inspection import permutation_importance

from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


#%% get data from artifacts

df_ports = pd.read_parquet(get_one_file_from_artifact('ports_features:latest').filepath)

df_centr = pd.read_parquet(get_one_file_from_artifact('centr_ports:latest').filepath)


df_merge = df_centr.reset_index().merge(df_ports, how="left", left_on="index", right_on="INDEX_NO")

all_feat = df_merge[df_ports.columns].drop(columns=["PORT_NAME", "Unnamed: 0", "REGION_NO"])

feature_names = [col for col in all_feat.columns]


#%%

yname = "page_rank_w_log_trips"
yname = "page_rank_w_trips"
cv_n_folds=5


all_Y = df_merge[df_centr.columns]

y = all_Y[yname]

# X = OneHotEncoder().fit_transform(all_feat)
X = OrdinalEncoder().fit_transform(all_feat)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# pipe.fit(X_train, y_train)
# pipe.score(X_test, y_test)

scoring = {'cv_r2': make_scorer(sklearn.metrics.r2_score), 'cv_neg_mean_absolute_error': 'neg_mean_absolute_error', 'cv_neg_mean_squared_error': 'neg_mean_squared_error'}

#%%

pipe = Pipeline(steps=[
    ('feature_selection', SelectKBest(mutual_info_regression, k=55)), 
    # ('feature_selection', SelectFromModel(RandomForestRegressor())), 
    # ('regression', SVR())
    ('regression', RandomForestRegressor())
])

start_time = time.time()
score_res = sklearn.model_selection.cross_validate(pipe, X_train, np.log(y_train), cv=cv_n_folds, scoring=scoring, n_jobs=10)
print({"time_cv_scoring": time.time() - start_time})
print(score_res["test_cv_r2"])


# %%

pipe = Pipeline(steps=[
    ('make_class', SelectKBest(mutual_info_regression, k=55)), 
    ('feature_selection', SelectKBest(mutual_info_regression, k=55)), 
    # ('feature_selection', SelectFromModel(RandomForestRegressor())), 
    # ('regression', SVR())
    ('regression', RandomForestRegressor())
])

start_time = time.time()
score_res = sklearn.model_selection.cross_validate(pipe, X_train, np.log(y_train), cv=cv_n_folds, scoring=scoring, n_jobs=10)
print({"time_cv_scoring": time.time() - start_time})
print(score_res["test_cv_r2"])


# %%
