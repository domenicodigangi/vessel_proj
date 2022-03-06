
#%%
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from vessel_proj.preprocess_data import get_data_path
import geopandas as gpd
from prefect import task
import requests
import zipfile


# %%

   
def download_and_read_wpi_data():
    raw_data_path = get_data_path() / "raw"
    # data from  https://msi.nga.mil/api/publications/download?key=16694622/SFH00000/WPI_Shapefile.zip

    url = ' https://msi.nga.mil/api/publications/download?key=16694622/SFH00000/WPI_Shapefile.zip'

    r = requests.get(url, allow_redirects=True)

    zip_path = raw_data_path / 'WPI_Shapefile.zip'

    with open(zip_path, 'wb') as file:
        file.write(r.content)
    
    with zipfile.ZipFile(zip_path, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(raw_data_path)
    
    df = pd.DataFrame(gpd.read_file(raw_data_path / 'WPI.shp' ).drop(columns="geometry"))

    df.to_parquet( get_data_path() / "interim"  /  "wpi_2019.parquet")
         

def clean_ports_info(df_ports):

    df_ports.describe().transpose()
    df_ports.describe(include = ["object"]).transpose()

    col_to_drop = ["CHART", "LAT_DEG", "LAT_MIN", "LONG_DEG", "LONG_MIN", "LAT_HEMI", "LONG_HEMI"]

    col_single_val = df_ports.columns[df_ports.apply(lambda x: pd.unique(x).shape[0]) == 1].values.tolist()
    print(col_single_val)
    
    col_to_drop.extend(col_single_val)

    df_ports.drop(columns=col_to_drop, inplace=True)
    
    
    for c in df_ports.select_dtypes(include=['object']).columns:
        df_ports[c] = df_ports[c].astype('category')
    

    df_ports["INDEX_NO"] = df_ports["INDEX_NO"].astype('int')
    df_ports = df_ports.set_index("INDEX_NO")

    # %% EDA ports info
    if False:
        for col in df_ports.columns:
            if col not in ["PORT_NAME", "COUNTRY"]:
                plt.figure()
                df_ports[col].hist()
                plt.title(col)
                plt.show()

    return df_ports
#%%

@task
def main(load_path, save_path):
    """
    load world port index info, cast types, drop some columns and store as parquet
    """
    df = pd.read_parquet(load_path /  "wpi_2019.parquet")

    df_clean = clean_ports_info(df)

    df_clean.to_parquet(save_path / 'ports_features.parquet')

if __name__ == "__main__":
    main()