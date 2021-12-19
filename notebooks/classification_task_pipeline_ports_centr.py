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
from sklearn import metrics

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
import sage
import argparse

logger = logging.getLogger(__file__)
    
#%% load data from artifacts

if False:
    # variables for dev
    yname = "page_rank_w_log_trips"
    model = RandomForestRegressor(random_state=0)
    run_sage = True
    n_sage_perm = None
    cv_n_folds = 5
    sage_imputer = "DefaultImputer"
    run = wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="regression_task", reinit=True, name="test_run")

def main(model, yname, run_sage, n_sage_perm, cv_n_folds, sage_imputer):

    logger.info(f"Running {model} {yname} {run_sage} {n_sage_perm} {cv_n_folds}")

    with wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="regression_task", reinit=True) as run:
    
        model_name = type(model).__name__

        wandb.log({"model": model_name, "var_predicted": yname, "cv_n_folds": cv_n_folds})

        df_ports = pd.read_parquet(get_one_file_from_artifact('ports_features:latest').filepath)

        df_centr = pd.read_parquet(get_one_file_from_artifact('centr_ports:latest').filepath)

        # add log of centralities to the list of centralities
        for c in df_centr.columns:
            df_centr[f"log_{c}"] = np.log(df_centr[c].values)

        df_centr.fillna(df_centr.min(skipna=True), inplace=True)
            
        df_merge = df_centr.reset_index().merge(df_ports, how="left", left_on="index", right_on="INDEX_NO")

        X = df_merge[df_ports.columns].drop(columns=["PORT_NAME", "Unnamed: 0", "REGION_NO"])

        feature_names = [col for col in X.columns]

        feat_names_non_cat = ["TIDE_RANGE", "LATITUDE", "LONGITUDE"]
        feat_names = list(X.columns)
        feat_names_cat = [f for f in feat_names if f not in feat_names_non_cat]

        wandb.log({"feat_names_non_cat": feat_names_non_cat, "feat_names_cat": feat_names_cat})

        # le = preprocessing.LabelEncoder()
        le = preprocessing.OrdinalEncoder()
        for col in X.columns:
            if col in feat_names_cat:
                X[col] = le.fit_transform(X[col].values.reshape(-1, 1))

        all_Y = df_merge[df_centr.columns]

        y = all_Y[yname]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        wandb.log({
            "train_set_size": X_train.shape[0],
            "test_set_size": X_test.shape[0],
            })

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train) 
        y_pred_test = model.predict(X_test) 

        score_funs = {
            'r2': metrics.r2_score,
            'neg_mean_absolute_error': metrics.mean_absolute_error,
            'mean_squared_error': metrics.mean_squared_error,
            'mean_absolute_percentage_error': metrics.mean_absolute_percentage_error
            }

        test_score = {f"test_set_{k}": f(y_test, y_pred_test) for k, f in score_funs.items()}
        wandb.log(test_score)
        train_score = {f"train_set_{k}": f(y_train, y_pred_train) for k, f in score_funs.items()}
        wandb.log(train_score)

        # run cross val on train set
        scoring = {k: make_scorer(f) for k,f in score_funs.items()}
        start_time = time.time()
        score_res = sklearn.model_selection.cross_validate(model, X_train, y_train, cv=cv_n_folds, scoring=scoring, n_jobs=10)
        wandb.log({"time_cv_scoring": time.time() - start_time})

        wandb.log({f"cv_{k}": v  for k, v in score_res.items()})

        # wandb.sklearn.plot_regressor(model, X_train, X_test, y_train, y_test, model_name=model_name)

        # %% Permutation feature importance from sklearn
        start_time = time.time()
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=20)

        feat_importances = pd.Series(result.importances_mean, index=feature_names).to_frame()
        wandb.log({"permutation_feat_importances": feat_importances.to_dict()})

        fig, ax = plt.subplots()
        feat_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        ax.set_xticklabels(feat_names)
        fig.tight_layout()
        wandb.log({"permutation_importances_plot": wandb.Image(fig)})


        wandb.log({"sage_importances_flag": str(run_sage)})

        if run_sage in ["True", "Y", "T"]:
            # experiment with sage, can be slow
            wandb.log({"n_sage_perm": n_sage_perm})
            wandb.log({"sage_imputer": sage_imputer})
            max_feat_sage = X_train.shape[1]

            feature_names = X_train.columns[:max_feat_sage].values

            train_size = 1024
            X_sage, _, y_sage, _ = train_test_split(X_train.iloc[:, :max_feat_sage].values, y_train.values, random_state=42, train_size = train_size)

            model.fit(X_sage[:, :max_feat_sage], y_sage)
            # Set up an imputer to handle missing features
            if sage_imputer == "MarginalImputer":
                imputer = sage.MarginalImputer(model, X_sage[:, :max_feat_sage])
            elif sage_imputer == "DefaultImputer":
                imputer = sage.DefaultImputer(model, np.zeros(max_feat_sage))
            
            # Set up an estimator
            estimator = sage.PermutationEstimator(imputer, 'mse')

            # Calculate SAGE values
            sage_values = estimator(X_test.values[:, :max_feat_sage], y_test.values, verbose=True, n_permutations=n_sage_perm)
            fig = sage_values.plot(feature_names, return_fig=True)
            [l.set_fontsize(8) for l in fig.axes[0].get_yticklabels()]
            
            wandb.log({"sage_importances_plot": wandb.Image(fig)})

            # Feature importance from SAGE
            start_time = time.time()
            wandb.log({"time_sage_feat_imp": time.time() - start_time})



#%%

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_sage", nargs='?', default="False",
                        help="compute and log sage feat importance")
    parser.add_argument("--n_sage_perm", nargs='?', default=1000000, type=int,
                        help="Maximum number of permutations in sage. If null it goes on until convergence")
    parser.add_argument("--sage_imputer", nargs='?', default="DefaultImputer",
                        help="compute and log sage feat importance")
    # parser.add_argument("cv_n_folds", default=5)
    args = parser.parse_args()

    print(args)


    all_models = [RandomForestRegressor(random_state=0), LinearRegression()]
    all_y_names = ["page_rank_w_log_trips", "page_rank_bin","log_page_rank_w_log_trips", "log_page_rank_bin"]
    
    run_sage = args.run_sage
    sage_imputer = args.sage_imputer
    n_sage_perm = int(args.n_sage_perm)
    cv_n_folds = 5 #int(args.cv_n_folds)

    for model in all_models:
        for yname in all_y_names:
            logger.info(f"Running {model}, y = {yname}")
            main(model, yname, run_sage, n_sage_perm, cv_n_folds, sage_imputer)
            logger.info(f"Finished {model}, y = {yname}")
        
        

