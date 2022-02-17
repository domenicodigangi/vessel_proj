#%%
import pandas as pd 

from vessel_proj.data import get_data_path, get_project_root


#%%


def make_and_save_latex_tab_sage_imp(df, tab_name, col_names):
    df = df.dropna(axis=1)

    features = [c.replace("sage_mean_", "") for c in df.columns if c.startswith("sage_mean")]
    df_mean = df[["disc_strategy"] + [c for c in df.columns if c.startswith("sage_mean")]].rename(columns = {f"sage_mean_{n}": n for n in features})
    df_std = df[["disc_strategy"] + [c for c in df.columns if c.startswith("sage_std")]].rename(columns = {f"sage_std_{n}": n for n in features})

    features = df_mean.mean(axis=0).sort_values(ascending=False).index.to_list()

    df_tab = pd.DataFrame()

    for n in features:
        df_tab[n] = df_mean[n].apply(lambda x: "$ {:5.1g}".format(x)).str.cat(df_std[n].apply(lambda x: "{:5.1g} $".format(1.6*x)), sep=" \pm ")



    df_tab = df_tab.rename(columns={n: n.replace("_", "\\_") for n in features})


    tab_fold = get_project_root() / "reports" / "tables"

    with open(tab_fold / f"{tab_name}.txt", "wt") as f:
        df_tab = df_tab.transpose()
        df_tab.columns = col_names
        df_tab.to_latex(buf = f, escape = False)

# %% top 5, 10, 15 %
df = pd.read_csv(get_data_path() / "processed" / "sage_top_5_10_15_wandb_export_2022-02-16T09_59_39.163+01_00.csv")

make_and_save_latex_tab_sage_imp(df, "tab_top_pct", ["5 pct", "10 pct", "15 pct"])

#%% kmeans groups
df = pd.read_csv(get_data_path() / "processed" / "sage_kmeans_2_3_4_5_wandb_export_2022-02-17T10_54_56.603+01_00.csv")

make_and_save_latex_tab_sage_imp(df, "kmeans", ["2", "3", "4", "5"])

# %%


df.columns
# %%
