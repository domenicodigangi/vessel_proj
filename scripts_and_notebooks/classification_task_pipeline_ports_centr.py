import seaborn as sns

sns.set_style("whitegrid")
import string
from argh import arg
import argh
import wandb
from vessel_proj.ds_utils import (
    get_wandb_root_path,
)
from vessel_proj.preprocess_data import get_latest_port_data_task
import logging
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import warnings
from vessel_proj.task.utils_classification_task_pipeline_ports_centr import (
    logwandb,
    add_avg_centr,
    encode_features,
    encode_df_features,
    drop_missing_cols,
    select_and_discretize_target,
    split_X_y,
    simple_impute_cols,
    impute_missing,
    train_score_model,
    estimate_sage,
    estimate_shap,
)

_PROJECT_NAME = "ports-feature-importance"
warnings.filterwarnings("ignore", category=ConvergenceWarning)
sns.set_theme(style="whitegrid")
logger = logging.getLogger(__file__)

local_save_images = True
# % define variables for development
if False:
    vessel_category = "cargo"
    feat_names_non_cat = ["TIDE_RANGE", "LATITUDE", "LONGITUDE"]
    cols_to_drop = ["PORT_NAME", "REGION_NO", "PUB"]
    yname = "avg_rank_centr"
    model_name = "RandomForestClassifier(random_state=0)"
    run_sage_and_shap = True
    n_sage_perm = None
    cv_n_folds = 5
    sage_imputer = "DefaultImputer"
    wandb.init(
        project=_PROJECT_NAME,
        dir=get_wandb_root_path(),
        group="classification_task",
        reinit=True,
        name="test_run",
        tags=["test_run"],
    )
    imputer_missing = "IterativeImputer()"  # "SimpleImputer()"
    test_run_flag = True
    # disc_strategy = "kmeans_3"
    disc_strategy = "top_10"
    log_of_target = False
    miss_threshold = 0.5

    one_run(
        model_name,
        yname,
        imputer_missing,
        run_sage_and_shap,
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


def one_run(
    vessel_category,
    model_name,
    yname,
    imputer_missing,
    run_sage_and_shap,
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
        project=_PROJECT_NAME,
        dir=get_wandb_root_path(),
        group="classification_task",
        reinit=True,
    ) as run:

        if test_run_flag:
            run.tags = (*run.tags, "test_run")

        run_pars = {
            "vessel_category": vessel_category,
            "model": model_name,
            "var_predicted": yname,
            "cv_n_folds": cv_n_folds,
            "disc_strategy": disc_strategy,
            "log_of_target": log_of_target,
            "cv_n_folds": cv_n_folds,
            "sage_imputer": sage_imputer,
            "run_sage_and_shap": run_sage_and_shap,
            "n_sage_perm": n_sage_perm,
            "miss_threshold": miss_threshold,
        }

        logwandb(run_pars)
        logger.info(f"RUNNING {run_pars}")

        data = get_latest_port_data_task.fn(vessel_category)

        data = add_avg_centr(data)

        data = encode_features(data, feat_names_non_cat, cols_to_drop)

        data = drop_missing_cols(data, threshold=miss_threshold)

        prep_X_y = select_and_discretize_target(
            data, yname, disc_strategy, log_of_target
        )

        train_test_X_y = split_X_y(prep_X_y)

        train_test_X_y = impute_missing(
            train_test_X_y, imputer_missing, feat_names_non_cat
        )

        train_score_model(train_test_X_y, model_name, cv_n_folds)

        if run_sage_and_shap in ["True", "Y", "T", True]:
            estimate_sage(train_test_X_y, model_name, sage_imputer, n_sage_perm)

            estimate_shap(data, yname, train_test_X_y, model_name)

        logger.info(f"FINISHED {run_pars}")


@arg("--run_sage_and_shap", help="compute and log sage feat importance")
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
    run_sage_and_shap=True,
    n_sage_perm=1000000,
    cv_n_folds=5,
    sage_imputer="DefaultImputer",
    disc_strategy="top_k",
    log_of_target=False,
    miss_threshold=0.5,
):
    all_vessel_category = ["cargo"]
    all_model_names = ["RandomForestClassifier(random_state=0)"]  # , "XGBClassifier()"]
    all_y_names = [
        # "page_rank_bin",
        # "page_rank_w_log_trips",
        # "closeness_bin",
        # "betweenness_bin",
        # "avg_rank_centr",
        "avg_centr",
    ]
    all_imputer_names = ["IterativeImputer()"]
    for vessel_category in all_vessel_category:
        for model_name in all_model_names:
            for yname in all_y_names:
                for imputer_missing in all_imputer_names:

                    if disc_strategy.startswith("top_"):
                        for k in [5, 10, 15]:  # 10
                            if disc_strategy == "top_k":
                                disc_strategy_run = f"top_{k}"
                            elif disc_strategy == "top_bottom_k":
                                disc_strategy_run = f"top_bottom_{k}"

                            one_run(
                                vessel_category,
                                model_name,
                                yname,
                                imputer_missing,
                                run_sage_and_shap,
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
                                vessel_category,
                                model_name,
                                yname,
                                imputer_missing,
                                run_sage_and_shap,
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
