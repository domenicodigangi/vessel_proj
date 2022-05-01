# %%
import seaborn as sns
from sklearn.cluster import KMeans
from vessel_proj.task.classification_task_pipeline_ports_centr import add_avg_centr
from vessel_proj.preprocess_data import get_latest_port_data_task
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
import pandas as pd
import numpy as np
from matplotlib import markers, pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn

from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.model_selection import train_test_split

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
sns.set_theme(style="darkgrid")

# %% get data from artifacts
vessel_category = "cargo"
data = get_latest_port_data_task.fn(vessel_category)

df_feat = data["features"]
df_centr = data["centralities"]
df_centr = add_avg_centr.fn(data)["centralities"]

df = df_centr.merge(df_feat, left_index=True, right_index=True)

df.sort_values(by="avg_rank_centr", ascending=False)[
    ["avg_centr", "avg_rank_centr", "PORT_NAME"]].head(50)

df_centr.sort_values(by="avg_rank_centr", ascending=False)


# %% Scatter
df = df_centr.drop(columns=[
                   "page_rank_w_trips", "centr_eig_w_trips", "centr_eig_bin", "centr_eig_w_log_trips"])


def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)


g = sns.PairGrid(df)
g.map_upper(sns.scatterplot)
g.map_lower(hexbin)
g.map_diag(sns.histplot)

# %%

g = sns.PairGrid(np.log(df))
g.map_upper(sns.scatterplot)
g.map_lower(hexbin)
g.map_diag(sns.histplot)


# %% number of clusters
centr = "page_rank_w_log_trips"
df = df_centr[[centr]]
Nc = range(1, 20)
kmeans_list = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans.fit(df).score(df) for kmeans in kmeans_list]
df_score = pd.DataFrame(data={"clusters": Nc, "score": score})
sns.relplot(data=df_score, x="clusters", y="score")

kmeans = KMeans(n_clusters=3, random_state=0).fit(df)
df['cluster'] = kmeans.labels_
sns.boxplot(df.cluster, df[centr])

# %%
df_merge = df_centr.reset_index().merge(
    df_feat, how="left", left_on="index", right_on="INDEX_NO")

all_feat = df_merge[df_feat.columns].drop(
    columns=["PORT_NAME", "Unnamed: 0", "REGION_NO"])

feature_names = [col for col in all_feat.columns]


df_merge.sort_values(by="page_rank_w_log_trips", ascending=False)[
    ["page_rank_w_log_trips", "PORT_NAME"]].head(20)

centr_name = "page_rank_bin"
df_merge.sort_values(by=centr_name, ascending=False)[
    [centr_name, "PORT_NAME"]].head(20)

# %%
df = df_merge[["page_rank_w_log_trips"]]

g = sns.PairGrid(df, diag_sharey=False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot)


# %%

yname = "page_rank_w_log_trips"
yname = "page_rank_w_trips"
cv_n_folds = 5


all_Y = df_merge[df_centr.columns]

y = all_Y[yname]

# X = OneHotEncoder().fit_transform(all_feat)
X = OrdinalEncoder().fit_transform(all_feat)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# pipe.fit(X_train, y_train)
# pipe.score(X_test, y_test)

scoring = {'cv_r2': make_scorer(sklearn.metrics.r2_score), 'cv_neg_mean_absolute_error':
           'neg_mean_absolute_error', 'cv_neg_mean_squared_error': 'neg_mean_squared_error'}

# %%

# %%
