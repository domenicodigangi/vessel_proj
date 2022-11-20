# %%
from typing import Dict, List, Optional

import seaborn as sns
from prefect import flow, task

sns.set_style("whitegrid")
import copy
import logging
import string
import time
import warnings

import numpy as np
import pandas as pd
import sage
import shap
import sklearn
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from vessel_proj.ds_utils import (
    catch_all_and_log,
)

import wandb

_PROJECT_NAME = "ports-feature-importance"
warnings.filterwarnings("ignore", category=ConvergenceWarning)


sns.set_theme(style="whitegrid")


logger = logging.getLogger(__file__)


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
        [
            "degree_in",
            "degree_out",
            "page_rank_bin",
            "page_rank_w_log_trips",
            "closeness_bin",
            "betweenness_bin",
        ]
    ]
    nports = df_centr.shape[0]
    df_centr["avg_rank_centr"] = df.rank().mean(axis=1) / nports
    df_centr["avg_centr"] = scaler.fit_transform(df).mean(axis=1)

    data["centralities"] = df_centr

    return data


@task
def encode_features(
    data_in: Dict,
    feat_names_non_cat: List[str],
    cols_to_drop: List[str],
):

    data = {k: v for k, v in data_in.items()}

    df = data["features"]

    df_X = df.drop(columns=cols_to_drop)
    feat_names = list(df_X.columns)
    feat_names_cat = [f for f in feat_names if f not in feat_names_non_cat]

    logwandb(
        {"feat_names_non_cat": feat_names_non_cat, "feat_names_cat": feat_names_cat}
    )

    df_X = encode_df_features(df_X, feat_names_cat)

    data["features"] = df_X
    data["dropped_cols"] = df[cols_to_drop]
    return data


def encode_df_features(df_X: pd.DataFrame, feat_names_cat: List[str]) -> pd.DataFrame:

    # Prepare the features
    # le = preprocessing.LabelEncoder()
    le = preprocessing.OrdinalEncoder()
    for col in df_X.columns:
        if col in feat_names_cat:
            df_X[col] = le.fit_transform(df_X[[col]])

    return df_X


@task
def drop_missing_cols(
    data_in,
    threshold=0.5,
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
    n_ports = df_centr.shape[0]

    df_merge = df_centr[[yname]].join(df_feat, how="left")

    X = df_merge[df_feat.columns]
    target = df_merge[[yname]].rename(columns={yname: "continuous"})
    if log_of_target:
        target["continuous_original"] = target["continuous"]
        target["continuous"] = np.log10(target["continuous"])

    if disc_strategy.startswith("top_"):
        n_top_pct = int(disc_strategy.split("_")[-1])

        n_top = int(np.round(n_ports * n_top_pct / 100))
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

    logwandb({"n_bins": n_bins, "n_top_ports": n_top})

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
        {
            "train_set_size": X_train.shape[0],
            "test_set_size": X_test.shape[0],
        }
    )

    return {"train": (X_train, y_train), "test": (X_test, y_test)}


def simple_impute_cols(
    feat_names_non_cat: List[str],
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
):
    for col in df_train.columns:
        if col in feat_names_non_cat:
            strategy = "mean"
        else:
            strategy = "most_frequent"

        imputer = SimpleImputer(strategy=strategy)

        df_train[col] = imputer.fit_transform(df_train[col].values.reshape(-1, 1))
        if df_test is not None:
            df_test[col] = imputer.transform(df_test[col].values.reshape(-1, 1))

    return df_train, df_test


@task
def impute_missing(train_test_X_y_in, imputer_missing, feat_names_non_cat):
    train_test_X_y = {k: copy.deepcopy(v) for k, v in train_test_X_y_in.items()}
    try:
        logwandb({"imputer_missing": imputer_missing})
    except:
        pass

    (X_train, y_train) = train_test_X_y["train"]
    (X_test, y_test) = train_test_X_y["test"]

    if imputer_missing.startswith("SimpleImputer"):
        X_train, X_test = simple_impute_cols(X_train, X_test)

    else:
        if imputer_missing.startswith("IterativeImputer"):
            imputer = IterativeImputer(initial_strategy="most_frequent")

        elif imputer_missing.startswith("KNNImputer"):
            imputer = KNNImputer()
        else:
            raise NotImplementedError("Imputer not considered")

        X_train.iloc[:, :] = imputer.fit_transform(X_train)
        X_test.iloc[:, :] = imputer.transform(X_test)

    train_test_X_y["train"] = (X_train, y_train)
    train_test_X_y["test"] = (X_test, y_test)

    return train_test_X_y


@task
def train_score_model(train_test_X_y_in, model_name, cv_n_folds):
    train_test_X_y = {k: copy.deepcopy(v) for k, v in train_test_X_y_in.items()}

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

    # % Permutation feature importance from sklearn
    start_time = time.time()
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=20
    )

    feat_importances = pd.Series(result.importances_mean, index=feat_names).to_frame()
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

            def yplot(x):
                return model.predict_proba(x)[:, 1]

        else:
            yplot = model.predict
        sns.set_style("whitegrid")
        fig, ax = plt.subplots()
        shap.plots.partial_dependence(
            feat_name,
            yplot,
            X_train,
            ice=False,
            model_expected_value=True,
            feature_expected_value=True,
            ax=ax,
            show=False,
        )
        if feat_name == "CARGODEPTH":
            alphabet_string = string.ascii_lowercase
            alphabet_list = list(alphabet_string)
            ticks = range(0, 16)
            labels = [alphabet_list[i] for i in ticks]
            plt.xticks(ticks, labels)
        elif feat_name == "HARBORSIZE":
            ticks = [0, 1, 2, 3]
            labels = ["L", "M", "S", "V"]
            plt.xticks(ticks, labels)

        logwandb({f"partial_dependence_{feat_name}": wandb.Image(fig)})


@task
def estimate_sage(train_test_X_y_in, model_name, sage_imputer, n_sage_perm):
    train_test_X_y = {k: copy.deepcopy(v) for k, v in train_test_X_y_in.items()}

    model = eval(model_name)
    (X_train, y_train) = train_test_X_y["train"]
    (X_test, y_test) = train_test_X_y["test"]
    feat_names = list(X_train.columns)
    # experiment with sage, can be slow

    model.fit(X_train.values, y_train.values)
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

    logwandb({f"sage_mean_{n}": v for n, v in zip(feat_names, sage_values.values)})
    logwandb({f"sage_std_{n}": v for n, v in zip(feat_names, sage_values.std)})
    logwandb({"sage_importances_plot": wandb.Image(fig)})

    # Feature importance from SAGE
    start_time = time.time()
    logwandb({"time_sage_feat_imp": time.time() - start_time})


@task
def estimate_shap(data_in, yname, train_test_X_y_in, model_name, n_ports_shap=100):
    data = {k: v for k, v in data_in.items()}
    train_test_X_y = {k: copy.deepcopy(v) for k, v in train_test_X_y_in.items()}

    start_time = time.time()

    model = eval(model_name)
    (X_train, y_train) = train_test_X_y["train"]
    (X_test, y_test) = train_test_X_y["test"]
    feat_names = list(X_train.columns)
    model.fit(X_train, y_train)
    # feat_name = "LONGITUDE"

    # compute the SHAP values for the linear model
    X_all = (
        pd.concat((X_train, X_test))
        .join(data["dropped_cols"][["PORT_NAME"]])
        .join(data["centralities"][[yname]])
        .sort_values(by=yname, ascending=False)
    )
    X_for_shap = X_all.drop(columns=["PORT_NAME", yname])

    explainer = shap.Explainer(model.predict, X_for_shap)
    shap_values = explainer(X_for_shap)
    df_shap = pd.DataFrame(
        shap_values.values, columns=shap_values.feature_names, index=X_for_shap.index
    ).join(X_all[["PORT_NAME", yname]], how="left")

    fig = plt.figure()
    shap.plots.beeswarm(shap_values, max_display=14)
    logwandb({"shap-swarm": wandb.Image(fig)})

    fig = plt.figure()
    shap.plots.bar(shap_values, max_display=14)
    logwandb({"shap-mean": wandb.Image(fig)})

    fig = plt.figure()
    shap.plots.heatmap(shap_values[:1000], max_display=14)
    logwandb({"shap-heat": wandb.Image(fig)})

    shap_tab = wandb.Table(dataframe=df_shap)

    logwandb({"shap_table": shap_tab})

    logwandb({"time_shap_feat_imp": time.time() - start_time})
