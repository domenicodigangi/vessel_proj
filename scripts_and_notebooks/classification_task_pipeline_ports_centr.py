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

from sklearn.metrics import make_scorer, plot_roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import sage
import argh
from argh import arg, expects_obj
from sklearn.preprocessing import StandardScaler
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
    run = wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="classification_task", reinit=True, name="test_run", tags=["test_run"])

    test_run_flag=True
    disc_strategy="kmeans_3"
    disc_strategy="top_100"
    log_of_target=False



#%%
def one_run(model, yname, run_sage, n_sage_perm, cv_n_folds, sage_imputer, disc_strategy, log_of_target: bool, test_run_flag=False):
    
    with wandb.init(project=get_project_name(), dir=get_wandb_root_path(), group="classification_task", reinit=True) as run:
        
        if test_run_flag:
            run.tags = (*run.tags, "test_run")

        model_name = type(model).__name__

        run_pars = {"model": model_name, "var_predicted": yname, "cv_n_folds": cv_n_folds, "disc_strategy": disc_strategy, "log_of_target": log_of_target, "cv_n_folds": cv_n_folds, "sage_imputer": sage_imputer, "run_sage": run_sage, "n_sage_perm": n_sage_perm} 

        wandb.log(run_pars)
        logger.info(f"RUNNING {run_pars}")

        data = get_latest_port_data()
        df_feat = data["features"]
        df_centr = data["centralities"]

        #add avg of different measures
        scaler = StandardScaler()
        df = df_centr[["page_rank_bin", "page_rank_w_log_trips", "closeness_bin", "betweenness_bin"]]
        df_centr["avg_centr"] = scaler.fit_transform(df).mean(axis=1)

        df_centr["avg_rank_centr"] = df.rank().mean(axis=1)

        df_merge = df_centr[yname].reset_index().merge(df_feat, how="left", left_on="index", right_on="INDEX_NO")

        X = df_merge[df_feat.columns].drop(columns=["PORT_NAME", "REGION_NO"])

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
                X[col] = le.fit_transform(X[[col]])

        my_imputer = SimpleImputer()
        X_imputed = my_imputer.fit_transform(X)
        X = pd.DataFrame(data=X_imputed, columns=X.columns)

        target = df_merge[[yname]].rename(columns={yname: "continuous"})
        if log_of_target:
            target["continuous_original"] = target["continuous"]
            target["continuous"] = np.log10(target["continuous"])

        if disc_strategy.startswith("top_"):
            n_top = int(disc_strategy.split("_")[-1])
            n_bins = 2
            target.sort_values(by="continuous", ascending=False, inplace=True)
            target["discrete"] = 0
            target["discrete"].iloc[:n_top] = 1
            if len(disc_strategy.split("_")) == 3:
                if disc_strategy.split("_")[1] == "bottom":
                    #keep only the top and bottom k observations (hence 2k in total)
                    inds_to_drop = target.index[n_top:-n_top]
                    target = target.drop(inds_to_drop)
                    X = X.drop(inds_to_drop)
        elif disc_strategy.startswith("kmeans_"):
            n_bins = int(disc_strategy.split("_")[-1])
            target["discrete"] = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans").fit_transform(target[["continuous"]])
        else:
            raise Exception()

        wandb.log({"n_bins": n_bins})
        
        fig, ax = plt.subplots()
        g=sns.boxplot(target["discrete"], target["continuous"], ax=ax)
        # Calculate number of obs per group & median to position labels
        nobs = [f"n obs {int(k)}: {v}"  for k, v in target['discrete'].value_counts().iteritems()]
        plt.legend(nobs)
        wandb.log({"boxplots_clusters": wandb.Image(fig)})


        y = target["discrete"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

        wandb.log({
            "train_set_size": X_train.shape[0],
            "test_set_size": X_test.shape[0],
            })

        score_funs = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        labels = list(range(n_bins))
        wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)

        if n_bins == 2:
            score_funs.append("roc_auc")
            fig, ax = plt.subplots()
            plot_roc_curve(model, X_test, y_test, ax=ax)
            wandb.log({"ROC Curve": wandb.Image(fig)})

        # run cross val on train set
        start_time = time.time()
        score_res = sklearn.model_selection.cross_validate(model, X_train, y_train, cv=cv_n_folds, scoring=score_funs, n_jobs=10)

        wandb.log({f"cv_{k}": v  for k, v in score_res.items()})
        wandb.log({f"avg_cv_{k}": np.mean(v)  for k, v in score_res.items()})
        

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

        if run_sage in ["True", "Y", "T", True]:
            # experiment with sage, can be slow

            model.fit(X_train, y_train)
            # Set up an imputer to handle missing features
            if sage_imputer == "MarginalImputer":
                imputer = sage.MarginalImputer(model, X_train)
            elif sage_imputer == "DefaultImputer":
                imputer = sage.DefaultImputer(model, np.zeros(X_train.shape[1]))

            # Set up an estimator
            estimator = sage.PermutationEstimator(imputer, 'cross entropy')

            # Calculate SAGE values
            sage_values = estimator(X_test.values, y_test.values, verbose=True, n_permutations=n_sage_perm)
            fig = sage_values.plot(feature_names, return_fig=True)
            [l.set_fontsize(8) for l in fig.axes[0].get_yticklabels()]

            wandb.log({f"sage_mean_{n}": v for n, v in zip(feature_names, sage_values.values) })
            wandb.log({f"sage_std_{n}": v for n, v in zip(feature_names, sage_values.std) })
            wandb.log({"sage_importances_plot": wandb.Image(fig)})

            # Feature importance from SAGE
            start_time = time.time()
            wandb.log({"time_sage_feat_imp": time.time() - start_time})

        logger.info(f"FINISHED {run_pars}")

#%%
@arg("--test_run_flag", help="tag as test run")
@arg("--run_sage", help="compute and log sage feat importance")
@arg("--n_sage_perm", help="Maximum number of permutations in sage. If null it goes on until convergence")
@arg("--cv_n_folds", help="N. Cross Val folds")
@arg("--sage_imputer", help="compute and log sage feat importance")
@arg("--disc_strategy", help="How are we going to define bins? top_100 (any number instead of 100), or kmeans https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-discretization")
def main(test_run_flag=False, run_sage=True, n_sage_perm=1000000, cv_n_folds=5, sage_imputer="DefaultImputer", disc_strategy="kmeans", log_of_target=False, njobs=4):

    all_models = [RandomForestClassifier(random_state=0), XGBClassifier()]
    all_y_names = ["page_rank_bin", "page_rank_w_log_trips", "closeness_bin", "betweenness_bin", "avg_rank_centr"]
  
    for model in all_models:
        for yname in all_y_names:

            if disc_strategy.startswith("top_"):
                for k in [50, 100, 250, 500]:
                    if disc_strategy == "top_k":
                        disc_strategy_run = f"top_{k}"
                    elif disc_strategy == "top_bottom_k":
                        disc_strategy_run = f"top_bottom_{k}"

                    one_run(model, yname, run_sage, n_sage_perm, cv_n_folds, sage_imputer, disc_strategy_run, log_of_target, test_run_flag=test_run_flag)

            elif disc_strategy == "kmeans":
                for n_bins in [2, 3, 4, 5]:
                    disc_strategy_run = f"kmeans_{n_bins}"
                    one_run(model, yname, run_sage, n_sage_perm, cv_n_folds, sage_imputer,  disc_strategy_run, log_of_target, test_run_flag=test_run_flag)

            
            
parser = argh.ArghParser()
parser.set_default_command(main)

if __name__ == "__main__":
    parser.dispatch()
    
        

