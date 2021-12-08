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

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from src.data import get_one_file_from_artifact, get_project_name, get_wandb_root_path
import wandb
from sklearn.inspection import permutation_importance

from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
logger = logging.getLogger(__file__)
    
#%% load data from artifacts

all_models = [RandomForestRegressor(random_state=0), XGBRegressor(), SVR(), LinearRegression(), Ridge()]

all_y_names = ["page_rank_w_trips", "page_rank_w_log_trips", "page_rank_bin", "centr_eig_bin"]

yname = "page_rank_w_log_trips"
model = RandomForestRegressor(random_state=0)

def main(model, yname):
    model_name = type(model).__name__
    
    cv_n_folds = 5

    with wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="regression_task", reinit=True) as run:

        wandb.log({"model": model_name, "var_predicted": yname, "cv_n_folds": cv_n_folds})
        
        df_ports = pd.read_parquet(get_one_file_from_artifact('ports_features:latest').filepath)

        df_centr = pd.read_parquet(get_one_file_from_artifact('centr_ports:latest').filepath)


        df_merge = df_centr.reset_index().merge(df_ports, how="left", left_on="index", right_on="INDEX_NO")

        X = df_merge[df_ports.columns].drop(columns=["PORT_NAME", "Unnamed: 0", "REGION_NO"])

        feature_names = [col for col in X.columns]

        

        feat_names_non_cat = ["TIDE_RANGE", "LATITUDE", "LONGITUDE"]
        feat_names = list(X.columns)
        feat_names_cat = [f for f in feat_names if f not in feat_names_non_cat]

        wandb.log({"feat_names_non_cat": feat_names_non_cat, "feat_names_cat": feat_names_cat})

        le = preprocessing.LabelEncoder()
        for col in X.columns:
            if col in feat_names_cat:
                X[col] = le.fit_transform(X[col])

        all_Y = df_merge[df_centr.columns]

        y = all_Y[yname]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        model.fit(X_train, y_train)
        wandb.log({"score_train_set": model.score(X_train, y_train)})
        wandb.log({"score_test_set": model.score(X_test, y_test)})

        scoring = {'cv_r2': make_scorer(sklearn.metrics.r2_score), 'cv_neg_mean_absolute_error': 'neg_mean_absolute_error', 'cv_neg_mean_squared_error': 'neg_mean_squared_error'}

        start_time = time.time()
        score_res = sklearn.model_selection.cross_validate(model, X_train, y_train, cv=cv_n_folds, scoring=scoring, n_jobs=10)
        wandb.log({"time_cv_scoring": time.time() - start_time})

        wandb.log(score_res)

        wandb.sklearn.plot_regressor(model, X_train, X_test, y_train, y_test, model_name=model_name)
        # %% Feature importance based on mean decrease in impurity

        
        start_time = time.time()
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=20)
        wandb.log({"time_permutation_importance": time.time() - start_time})
        
        feat_importances = pd.Series(result.importances_mean, index=feature_names)
        wandb.log({"permutation_feat_importances": feat_importances.to_frame().to_dict()})

        fig, ax = plt.subplots()
        feat_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        ax.set_xticklabels(feat_names)
        fig.tight_layout()

        wandb.log({"permutation_importances_plot": fig})

#%%

if __name__ == "__main__":
    for model in all_models:
        for yname in all_y_names:
            try:
                logger.info(f"Running {model}, y = {yname}")
                main(model, yname)
                logger.info(f"Finished {model}, y = {yname}")
            except:
                logger.error(f"Failed {model}, y = {yname}")
                