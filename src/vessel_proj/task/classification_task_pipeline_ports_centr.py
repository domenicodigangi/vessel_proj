# %%
from sklearn.base import ClassifierMixin
from prefect import task, flow
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from argh import arg
import argh
import sage
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import make_scorer, plot_roc_curve
from sklearn.inspection import permutation_importance
import wandb
import shap
from vessel_proj.utils import catch_all_and_log
from vessel_proj.preprocess_data import (
    get_project_name,
    get_wandb_entity,
    get_wandb_root_path,
    get_latest_port_data_task,
)
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import logging
import time
import copy
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


sns.set_theme(style="darkgrid")


logger = logging.getLogger(__file__)


# %%


@catch_all_and_log
def logwandb(*args, **kwargs):
    wandb.log(*args, **kwargs)


@task
def add_avg_centr(data_in):
    data = {k: v for k, v in data_in.items()}
    df_centr = data["centralities"]
    # add avg of different measures
    scaler = StandardScaler()
    df = df_centr[
        ["page_rank_bin", "page_rank_w_log_trips",
            "closeness_bin", "betweenness_bin"]
    ]
    df_centr["avg_centr"] = scaler.fit_transform(df).mean(axis=1)
    df_centr["avg_rank_centr"] = df.rank().mean(axis=1)

    data["centralities"] = df_centr

    return data


@task
def encode_features(
    data_in, feat_names_non_cat, cols_to_drop,
):

    data = {k: v for k, v in data_in.items()}

    df = data["features"]

    X = df.drop(columns=cols_to_drop)

    feat_names = list(X.columns)
    feat_names_cat = [f for f in feat_names if f not in feat_names_non_cat]

    logwandb(
        {"feat_names_non_cat": feat_names_non_cat, "feat_names_cat": feat_names_cat}
    )

    # Prepare the features
    # le = preprocessing.LabelEncoder()
    le = preprocessing.OrdinalEncoder()
    for col in X.columns:
        if col in feat_names_cat:
            X[col] = le.fit_transform(X[[col]])

    data["features"] = X
    return data


@task
def drop_missing_cols(
    data_in, threshold=0.5,
):
    data = {k: v for k, v in data_in.items()}
    df = data["features"]

    fract_miss = df.isnull().mean().sort_values(ascending=False)
    cols_to_drop = fract_miss.index[fract_miss > threshold]

    logger.info(f"Dropping {cols_to_drop.shape[0]} columns: {cols_to_drop}")
    data["features"] = df.drop(columns=cols_to_drop)

    return data


@task
def select_and_discretize_target(data_in, yname, disc_strategy, log_of_target):
    data = {k: v for k, v in data_in.items()}
    df_feat = data["features"]
    df_centr = data["centralities"]

    df_merge = (
        df_centr[yname]
        .reset_index()
        .merge(df_feat, how="left", left_on="index", right_on="INDEX_NO")
    )

    X = df_merge[df_feat.columns]

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
                # keep only the top and bottom k observations (hence 2k in total)
                inds_to_drop = target.index[n_top:-n_top]
                target = target.drop(inds_to_drop)
                X = X.drop(inds_to_drop)
    elif disc_strategy.startswith("kmeans_"):
        n_bins = int(disc_strategy.split("_")[-1])
        target["discrete"] = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="kmeans"
        ).fit_transform(target[["continuous"]])
    else:
        raise Exception()

    logwandb({"n_bins": n_bins})

    fig, ax = plt.subplots()
    g = sns.boxplot(target["discrete"], target["continuous"], ax=ax)
    # Calculate number of obs per group & median to position labels
    nobs = [
        f"n obs {int(k)}: {v}" for k, v in target["discrete"].value_counts().iteritems()
    ]
    plt.legend(nobs)
    logwandb({"boxplots_clusters": wandb.Image(fig)})
    y = target["discrete"]

    X_y = (X, y)
    return X_y


@task
def split_X_y(X_y):
    X, y = X_y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y
    )

    logwandb(
        {"train_set_size": X_train.shape[0],
            "test_set_size": X_test.shape[0], }
    )

    return {"train": (X_train, y_train), "test": (X_test, y_test)}


@task
def impute_missing(train_test_X_y_in, imputer_missing, feat_names_non_cat):
    train_test_X_y = {k: copy.deepcopy(v)
                      for k, v in train_test_X_y_in.items()}
    try:
        logwandb({"imputer_missing": imputer_missing})
    except:
        pass

    (X_train, y_train) = train_test_X_y["train"]
    (X_test, y_test) = train_test_X_y["test"]

    if imputer_missing.startswith("SimpleImputer"):
        for col in X_train.columns:
            if col in feat_names_non_cat:
                strategy = "mean"
            else:
                strategy = "most_frequent"

            imputer = SimpleImputer(strategy=strategy)

            X_train[col] = imputer.fit_transform(
                X_train[col].values.reshape(-1, 1))
            X_test[col] = imputer.transform(X_test[col].values.reshape(-1, 1))

    elif imputer_missing.startswith("IterativeImputer"):
        imputer = IterativeImputer(initial_strategy="most_frequent")
        for col in X_train.columns:
            X_train.values = imputer.fit_transform(X_train)
            X_test.values = imputer.transform(X_test)

    elif imputer_missing.startswith("KNNImputer"):
        imputer = KNNImputer()
        X_train.values = imputer.fit_transform(X_train)
        X_test.values = imputer.transform(X_test)

    train_test_X_y["train"] = (X_train, y_train)
    train_test_X_y["test"] = (X_test, y_test)

    return train_test_X_y


@task
def train_score_model(train_test_X_y_in, model_name, cv_n_folds):
    train_test_X_y = {k: copy.deepcopy(v)
                      for k, v in train_test_X_y_in.items()}

    model = eval(model_name)
    (X_train, y_train) = train_test_X_y["train"]
    (X_test, y_test) = train_test_X_y["test"]

    n_bins = y_train.unique().shape[0]

    feat_names = list(X_train.columns)

    score_funs = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    labels = list(range(n_bins))
    wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)

    if n_bins == 2:
        score_funs.append("roc_auc")
        fig, ax = plt.subplots()
        plot_roc_curve(model, X_test, y_test, ax=ax)
        logwandb({"ROC Curve": wandb.Image(fig)})

    # run cross val on train set
    start_time = time.time()
    score_res = sklearn.model_selection.cross_validate(
        model, X_train, y_train, cv=cv_n_folds, scoring=score_funs, n_jobs=10
    )

    logwandb({f"cv_{k}": v for k, v in score_res.items()})
    logwandb({f"avg_cv_{k}": np.mean(v) for k, v in score_res.items()})

    # %% Permutation feature importance from sklearn
    start_time = time.time()
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=20
    )

    feat_importances = pd.Series(
        result.importances_mean, index=feat_names).to_frame()
    logwandb({"permutation_feat_importances": feat_importances.to_dict()})

    fig, ax = plt.subplots()
    feat_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    ax.set_xticklabels(feat_names)
    fig.tight_layout()
    logwandb({"permutation_importances_plot": wandb.Image(fig)})

    for feat_name in X_train.columns:
        n_bins = y_train.unique().shape[0]
        if n_bins == 2:
            def yplot(x): return model.predict_proba(x)[:, 1]
        else:
            yplot = model.predict
        fig, ax = plt.subplots()
        shap.plots.partial_dependence(
            feat_name, yplot, X_train, ice=False,
            model_expected_value=True, feature_expected_value=True, ax=ax
        )

        logwandb({f"partial_dependence_{feat_name}": wandb.Image(fig)})



@task
def estimate_sage(train_test_X_y_in, model_name, sage_imputer, n_sage_perm):
    train_test_X_y = {k: copy.deepcopy(v)
                      for k, v in train_test_X_y_in.items()}

    model = eval(model_name)
    (X_train, y_train) = train_test_X_y["train"]
    (X_test, y_test) = train_test_X_y["test"]
    feat_names = list(X_train.columns)
    # experiment with sage, can be slow

    model.fit(X_train, y_train)
    # Set up an imputer to handle missing features
    if sage_imputer == "MarginalImputer":
        imputer = sage.MarginalImputer(model, X_train)
    elif sage_imputer == "DefaultImputer":
        imputer = sage.DefaultImputer(model, np.zeros(X_train.shape[1]))

    # Set up an estimator
    estimator = sage.PermutationEstimator(imputer, "cross entropy")

    # Calculate SAGE values
    sage_values = estimator(
        X_test.values, y_test.values, verbose=True, n_permutations=n_sage_perm
    )
    fig = sage_values.plot(feat_names, return_fig=True)
    [l.set_fontsize(8) for l in fig.axes[0].get_yticklabels()]

    logwandb({f"sage_mean_{n}": v for n, v in zip(
        feat_names, sage_values.values)})
    logwandb({f"sage_std_{n}": v for n, v in zip(feat_names, sage_values.std)})
    logwandb({"sage_importances_plot": wandb.Image(fig)})

    # Feature importance from SAGE
    start_time = time.time()
    logwandb({"time_sage_feat_imp": time.time() - start_time})


@task
def estimate_shap(train_test_X_y_in, model_name, sage_imputer, n_sage_perm):
    train_test_X_y = {k: copy.deepcopy(v)
                      for k, v in train_test_X_y_in.items()}

    model = eval(model_name)
    (X_train, y_train) = train_test_X_y["train"]
    (X_test, y_test) = train_test_X_y["test"]
    feat_names = list(X_train.columns)
    model.fit(X_train, y_train)
    # feat_name = "LONGITUDE"
   
    # compute the SHAP values for the linear model
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train)
    shap.plots.beeswarm(shap_values, max_display=14)

    

    

    # Set up an imputer to handle missing features
    if sage_imputer == "MarginalImputer":
        imputer = sage.MarginalImputer(model, X_train)
    elif sage_imputer == "DefaultImputer":
        imputer = sage.DefaultImputer(model, np.zeros(X_train.shape[1]))

    # Set up an estimator
    estimator = sage.PermutationEstimator(imputer, "cross entropy")

    # Calculate SAGE values
    sage_values = estimator(
        X_test.values, y_test.values, verbose=True, n_permutations=n_sage_perm
    )
    fig = sage_values.plot(feat_names, return_fig=True)
    [l.set_fontsize(8) for l in fig.axes[0].get_yticklabels()]

    logwandb({f"sage_mean_{n}": v for n, v in zip(
        feat_names, sage_values.values)})
    logwandb({f"sage_std_{n}": v for n, v in zip(feat_names, sage_values.std)})
    logwandb({"sage_importances_plot": wandb.Image(fig)})

    # Feature importance from SAGE
    start_time = time.time()
    logwandb({"time_sage_feat_imp": time.time() - start_time})


# %%
@flow
def one_run(
    model_name,
    yname,
    imputer_missing,
    run_sage,
    n_sage_perm,
    cv_n_folds,
    sage_imputer,
    disc_strategy,
    miss_threshold,
    log_of_target: bool,
    feat_names_non_cat=["TIDE_RANGE", "LATITUDE", "LONGITUDE"],
    cols_to_drop=["PORT_NAME", "REGION_NO", "PUB"],
    test_run_flag=False,
):

    with wandb.init(
        project=get_project_name(),
        dir=get_wandb_root_path(),
        group="classification_task",
        reinit=True,
    ) as run:

        if test_run_flag:
            run.tags = (*run.tags, "test_run")

        run_pars = {
            "model": model_name,
            "var_predicted": yname,
            "cv_n_folds": cv_n_folds,
            "disc_strategy": disc_strategy,
            "log_of_target": log_of_target,
            "cv_n_folds": cv_n_folds,
            "sage_imputer": sage_imputer,
            "run_sage": run_sage,
            "n_sage_perm": n_sage_perm,
            "miss_threshold": miss_threshold,
        }

        logwandb(run_pars)
        logger.info(f"RUNNING {run_pars}")

        data = get_latest_port_data_task.fn()

        data = add_avg_centr.fn(data)

        data = encode_features.fn(data, feat_names_non_cat, cols_to_drop)

        data = drop_missing_cols.fn(data, threshold=miss_threshold)

        prep_X_y = select_and_discretize_target.fn(
            data, yname, disc_strategy, log_of_target
        )

        train_test_X_y = split_X_y.fn(prep_X_y)

        train_test_X_y = impute_missing.fn(
            train_test_X_y, imputer_missing, feat_names_non_cat
        )

        train_score_model.fn(train_test_X_y, model_name, cv_n_folds)

        if run_sage in ["True", "Y", "T", True]:
            estimate_sage.fn(train_test_X_y, model_name,
                             sage_imputer, n_sage_perm)

        logger.info(f"FINISHED {run_pars}")


# %% define variable for development
if False:
    feat_names_non_cat = ["TIDE_RANGE", "LATITUDE", "LONGITUDE"]
    cols_to_drop = ["PORT_NAME", "REGION_NO", "PUB"]
    yname = "page_rank_w_log_trips"
    model_name = "RandomForestClassifier(random_state=0)"
    run_sage = True
    n_sage_perm = None
    cv_n_folds = 5
    sage_imputer = "DefaultImputer"
    wandb.init(
        project=get_project_name(),
        dir=get_wandb_root_path(),
        group="classification_task",
        reinit=True,
        name="test_run",
        tags=["test_run"],
    )
    imputer_missing = "SimpleImputer()"
    test_run_flag = True
    # disc_strategy = "kmeans_3"
    disc_strategy = "top_100"
    log_of_target = False
    miss_threshold = 0.5

    one_run(
        model_name,
        yname,
        imputer_missing,
        run_sage,
        n_sage_perm,
        cv_n_folds,
        sage_imputer,
        disc_strategy,
        miss_threshold,
        log_of_target,
        feat_names_non_cat=["TIDE_RANGE", "LATITUDE", "LONGITUDE"],
        cols_to_drop=["PORT_NAME", "REGION_NO", "PUB"],
        test_run_flag=False,
    )

# %%


@arg("--run_sage", help="compute and log sage feat importance")
@arg(
    "--n_sage_perm",
    help="Maximum number of permutations in sage. If null it goes on until convergence",
)
@arg("--cv_n_folds", help="N. Cross Val folds")
@arg("--sage_imputer", help="compute and log sage feat importance")
@arg(
    "--disc_strategy",
    help="How are we going to define bins? top_k% (any number instead of 100), or kmeans https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-discretization",
)
def main(
    run_sage=True,
    n_sage_perm=1000000,
    cv_n_folds=5,
    sage_imputer="DefaultImputer",
    disc_strategy="top_k",
    log_of_target=False,
    miss_threshold=0.5,
):

    all_model_names = ["RandomForestClassifier(random_state=0)", "XGBClassifier()"]
    all_y_names = [
        "page_rank_bin",
        "page_rank_w_log_trips",
        "closeness_bin",
        "betweenness_bin",
        "avg_rank_centr",
    ]
    all_imputer_names = ["SimpleImputer()", "KNNImputer()",
                         "IterativeImputer()"]
    for model_name in all_model_names:
        for yname in all_y_names:
            for imputer_missing in all_imputer_names:

                if disc_strategy.startswith("top_"):
                    for k in [182, 363, 544]:
                        if disc_strategy == "top_k":
                            disc_strategy_run = f"top_{k}"
                        elif disc_strategy == "top_bottom_k":
                            disc_strategy_run = f"top_bottom_{k}"

                        one_run(
                            model_name,
                            yname,
                            imputer_missing,
                            run_sage,
                            n_sage_perm,
                            cv_n_folds,
                            sage_imputer,
                            disc_strategy_run,
                            miss_threshold,
                            log_of_target,
                        )

                elif disc_strategy == "kmeans":
                    for n_bins in [2, 3, 4, 5]:
                        disc_strategy_run = f"kmeans_{n_bins}"
                        one_run(
                            model_name,
                            yname,
                            imputer_missing,
                            run_sage,
                            n_sage_perm,
                            cv_n_folds,
                            sage_imputer,
                            disc_strategy_run,
                            miss_threshold,
                            log_of_target,
                        )


parser = argh.ArghParser()
parser.set_default_command(main)

if __name__ == "__main__":
    parser.dispatch()


# %%
