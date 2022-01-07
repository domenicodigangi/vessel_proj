#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import logging
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from vessel_proj.data import get_one_file_from_artifact, get_project_name, get_wandb_root_path, get_latest_port_data

import wandb
from sklearn.inspection import permutation_importance

from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
from xgboost import XGBClassifier
import sage
import argh
from argh import arg, expects_obj
import seaborn as sns
sns.set_theme(style="darkgrid")

from joblib import Parallel, delayed

logger = logging.getLogger(__file__)
    
#%% define variable for development
if False:
    yname = "page_rank_w_log_trips"
    model = RandomForestClassifier(random_state=0)
    run_sage = True
    n_sage_perm = None
    cv_n_folds = 5
    sage_imputer = "DefaultImputer"
    run = wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="regression_task", reinit=True, name="test_run", tags=["test_run"])
    n_bins=3
    test_run_flag=True
    disc_strategy="kmeans"



#%%
def one_run(model, yname, run_sage, n_sage_perm, cv_n_folds, sage_imputer, n_bins, disc_strategy, test_run_flag=False):

    logger.info(f"Running {model} {yname} {run_sage} {n_sage_perm} {cv_n_folds}")

    with wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="classification_task", reinit=True) as run:
        
        if test_run_flag:
            run.tags = (*run.tags, "test_run")

        model_name = type(model).__name__

        wandb.log({"model": model_name, "var_predicted": yname, "cv_n_folds": cv_n_folds, "n_bins": n_bins, "disc_strategy": disc_strategy})

        data = get_latest_port_data()
        df_feat = data["features"]
        df_centr = data["centralities"]

        # add log of centralities to the list of centralities
        for c in df_centr.columns:
            df_centr[f"log_{c}"] = np.log(df_centr[c].values)

        df_centr.fillna(df_centr.min(skipna=True), inplace=True)
            
        df_merge = df_centr.reset_index().merge(df_feat, how="left", left_on="index", right_on="INDEX_NO")

        X = df_merge[df_feat.columns].drop(columns=["PORT_NAME", "Unnamed: 0", "REGION_NO"])

        feature_names = [col for col in X.columns]

        feat_names_non_cat = ["TIDE_RANGE", "LATITUDE", "LONGITUDE"]
        feat_names = list(X.columns)
        feat_names_cat = [f for f in feat_names if f not in feat_names_non_cat]

        wandb.log({"feat_names_non_cat": feat_names_non_cat, "feat_names_cat": feat_names_cat})

        # Prepare the features
        # le = preprocessing.LabelEncoder()
        le = preprocessing.OrdinalEncoder()
        for col in X.columns:
            if col in feat_names_cat:
                X[col] = le.fit_transform(X[col].values.reshape(-1, 1))

        all_Y = df_merge[df_centr.columns]

        y = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=disc_strategy).fit_transform(all_Y[[yname]]).ravel()
        
        df = df_centr[[yname]]
        df['cluster'] = y
        fig, ax = plt.subplots()
        g=sns.boxplot(df.cluster, df[yname], ax=ax)
        # Calculate number of obs per group & median to position labels
        nobs = [f"n obs {int(k)}: {v}"  for k, v in df['cluster'].value_counts().iteritems()]
        plt.legend(nobs)
        wandb.log({"boxplots_clusters": wandb.Image(fig)})

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        wandb.log({
            "train_set_size": X_train.shape[0],
            "test_set_size": X_test.shape[0],
            })

        score_funs = {
            'accuracy': metrics.accuracy_score}

        # run cross val on train set
        scoring = {k: make_scorer(f) for k,f in score_funs.items()}
        start_time = time.time()
        score_res = sklearn.model_selection.cross_validate(model, X_train, y_train, cv=cv_n_folds, scoring=scoring, n_jobs=10)
        
        wandb.log({f"cv_{k}": v  for k, v in score_res.items()})

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        labels = list(range(n_bins))
        wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)

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

        if run_sage in ["True", "Y", "T", True]:
            # experiment with sage, can be slow
            wandb.log({"n_sage_perm": n_sage_perm})
            wandb.log({"sage_imputer": sage_imputer})
           
            model.fit(X_train, y_train)
            # Set up an imputer to handle missing features
            if sage_imputer == "MarginalImputer":
                imputer = sage.MarginalImputer(model, X_train)
            elif sage_imputer == "DefaultImputer":
                imputer = sage.DefaultImputer(model, np.zeros(X_train.shape[1]))
            

            # Set up an estimator
            estimator = sage.PermutationEstimator(imputer, 'cross entropy')

            # Calculate SAGE values
            sage_values = estimator(X_test.values, y_test, verbose=True, n_permutations=n_sage_perm)
            fig = sage_values.plot(feature_names, return_fig=True)
            [l.set_fontsize(8) for l in fig.axes[0].get_yticklabels()]
            
            wandb.log({"sage_importances_plot": wandb.Image(fig)})

            # Feature importance from SAGE
            start_time = time.time()
            wandb.log({"time_sage_feat_imp": time.time() - start_time})

#%%
@arg("--test_run_flag", help="tag as test run")
@arg("--run_sage", help="compute and log sage feat importance")
@arg("--n_bins_min", help="Min Number of bins for the target variable")
@arg("--n_bins_max", help="Max Number of bins for the target variable")
@arg("--disc_strategy", help="How are we going to define bins? https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-discretization")
@arg("--n_sage_perm", help="Maximum number of permutations in sage. If null it goes on until convergence")
@arg("--cv_n_folds", help="N. Cross Val folds")
@arg("--sage_imputer", help="compute and log sage feat importance")
def main(test_run_flag=False, run_sage=True, n_sage_perm=1000000, n_bins_min=2, n_bins_max=6, cv_n_folds=5, sage_imputer="DefaultImputer", disc_strategy="kmeans", njobs=4):

    all_models = [RandomForestClassifier(random_state=0), XGBClassifier()]
    all_y_names = ["page_rank_w_log_trips", "page_rank_bin"]
  
    for model in all_models:
        for yname in all_y_names:
            
            logger.info(f"Running {model}, y = {yname}, {model} {yname} {run_sage} {n_sage_perm} {cv_n_folds} {sage_imputer} {n_bins_min} {n_bins_max} {test_run_flag}")
            
            for n_bins in range(n_bins_min, n_bins_max+1):
                one_run(model, yname, run_sage, n_sage_perm, cv_n_folds, sage_imputer, n_bins, disc_strategy, test_run_flag=test_run_flag)

            logger.info(f"Finished {model}, y = {yname}, {model} {yname} {run_sage} {n_sage_perm} {cv_n_folds} {sage_imputer} {n_bins_min} {n_bins_max} {test_run_flag}")
            
            
parser = argh.ArghParser()
parser.set_default_command(main)

if __name__ == "__main__":
    parser.dispatch()
    
        

