# %%
import wandb
import pandas as pd
from matplotlib import markers, pyplot as plt
from vessel_proj.ds_utils import get_latest_port_data_task, get_project_name, get_data_path, get_project_root
from vessel_proj.task.classification_task_pipeline_ports_centr import add_avg_centr
import seaborn as sns
sns.set_theme(style="darkgrid")

# %% get data from artifacts
vessel_category = "cargo"
data = get_latest_port_data_task.fn(vessel_category)

df_feat = data["features"]
df_centr = data["centralities"]
df_centr = add_avg_centr.fn(data)["centralities"]

# %% Distribution avg_rank
df = df_centr
for col in ["avg_centr", "avg_rank_centr"]:
    df_tab = (df
              .merge(df_feat, left_index=True, right_index=True)
              .sort_values(by=col, ascending=False)[
                  ["PORT_NAME", col]]
              .head(10)
              .reset_index()
              .drop(columns=["index"])
              )

    tab_fold = get_project_root() / "reports" / "tables"
    fig_fold = get_project_root() / "reports" / "figures"

    with open(tab_fold / f"{col}.txt", "wt") as f:
        df_tab.to_latex(buf=f, escape=False)

        plt.figure()
        ax = sns.histplot(df[col])
        for i, q in df[col].quantile([0.95, 0.9, 0.85]).iteritems():
            ax.axvline(q, color="r")
        fig = ax.get_figure()
        fig.savefig(fig_fold / f"hist{col}.png")


# %% Load SHAP run data

wandb.init()
tab_fold = get_project_root() / "reports" / "tables"

run_id = "n8xhfk0y"
tab = wandb.use_artifact(
    f"digangidomenico/ports-feat-importance/run-{run_id}-shap_table:v0").get("shap_table")

# %% Create and save SHAP tab
df_shap = pd.DataFrame([r for i, r in tab.iterrows()], columns=tab.columns)
df_shap = df_shap.rename(columns={df_shap.columns[-1]: "Centrality"})
# col_names = ["PORT_NAME", df_shap.columns[-1]]
# [col_names.extend([c for c in df_shap.columns if c not in col_names ])]

df_shap["Top_2_Feat_SHAP"] = df_shap.drop(columns=["PORT_NAME", "Centrality"]).apply(
    lambda s: ", ".join(s.abs().nlargest(2).index.tolist()).replace("_", "\\_"), axis=1)

# df_shap["min_shap"] = df_shap.drop(columns=["PORT_NAME", "Centrality", "max_shap"]).idxmin(axis=1)


col_names = ["PORT_NAME", "Centrality", "Top_2_Feat_SHAP"]

df_tab = (df_shap[:20][col_names]
          .round(4)
          .rename(columns={n: n.replace("_", "\\_") for n in col_names})
          )
tab_name = "shap_values"
with open(tab_fold / f"{tab_name}.txt", "wt") as f:
    df_tab.to_latex(buf=f, escape=False)


# %%


def make_and_save_latex_tab_sage_imp(df, tab_name, col_names):
    df = df.dropna(axis=1)

    features = [c.replace("sage_mean_", "")
                for c in df.columns if c.startswith("sage_mean")]
    df_mean = df[["disc_strategy"] + [c for c in df.columns if c.startswith(
        "sage_mean")]].rename(columns={f"sage_mean_{n}": n for n in features})
    # df_std = df[["disc_strategy"] + [c for c in df.columns if c.startswith(
    #     "sage_std")]].rename(columns={f"sage_std_{n}": n for n in features})

    features = df_mean.mean(axis=0).sort_values(
        ascending=False).index.to_list()

    df_tab = pd.DataFrame()

    for n in features:
        df_tab[n] = df_mean[n].apply(lambda x: "$ {:5.1g}".format(x))

        # df_tab[n] = df_mean[n].apply(lambda x: "$ {:5.1g}".format(x)).str.cat(
        #     df_std[n].apply(lambda x: "{:5.1g} $".format(1.6*x)), sep=" \pm ")

    df_tab = df_tab.rename(
        columns={n: n.replace("_", "\\_") for n in features})

    tab_fold = get_project_root() / "reports" / "tables"

    with open(tab_fold / f"{tab_name}.txt", "wt") as f:
        df_tab = df_tab.transpose()
        df_tab.columns = col_names
        df_tab.to_latex(buf=f, escape=False)


# %% top 5, 10, 15 %
df = pd.read_csv(get_data_path() / "processed" /
                 "sage_top_5_10_15_wandb_export_2022-04-13T17_59_52.595+02_00.csv")

make_and_save_latex_tab_sage_imp(
    df, "tab_top_pct", ["5 pct", "10 pct", "15 pct"])

# %% kmeans groups
df = pd.read_csv(get_data_path() / "processed" /
                 "sage_kmeans_2_3_4_5_wandb_export_2022-02-17T10_54_56.603+01_00.csv")

make_and_save_latex_tab_sage_imp(df, "kmeans", ["2", "3", "4", "5"])

# %%


df.columns
# %%
