from pathlib import Path
import logging
import pandas as pd
from tqdm import tqdm 
import logging
import wandb
from pathlib import Path
from types import SimpleNamespace



logger = logging.getLogger()
logger.setLevel(logging.INFO)



def create_data_fold_structure(project_dir):
    logger.info(f"Creating data folder structure in {project_dir}")
    Path.mkdir(project_dir / "data", exist_ok = True)
    Path.mkdir(project_dir / "data" / "raw", exist_ok = True)
    Path.mkdir(project_dir / "data" / "processed", exist_ok = True)
    Path.mkdir(project_dir / "data" / "interim", exist_ok = True)
    Path.mkdir(project_dir / "data" / "external", exist_ok = True)


def get_project_name():
    return "vessel-proj"

def get_project_root():
    return Path(__file__).parent.parent.parent.parent

def get_data_path():
    proj_root = get_project_root() 
    data_path = proj_root / "data"

    return data_path

def get_wandb_root_path():
    root_path = get_data_path() / "root_wandb"
    logging.warning(f"using hardcoded path for data in barbera: {root_path}" )

    return root_path


def get_artifacts_path():
    art_path = get_wandb_root_path() / "artifacts"
    art_path.mkdir(exist_ok=True)
    return art_path

def get_one_file_from_artifact(name, run=None, type=None):

    if run is not None:
        artifact = run.use_artifact(name, type=type)
    else:
        api = wandb.Api()
        fullname = f"{get_project_name()}/{name}"
        artifact = api.artifact(fullname, type=type)
        

    d = {"artifact": artifact}
        
    try:
        artifact_dir = artifact.download(root=get_wandb_root_path() / artifact._default_root())
        d["artifact_dir"] = artifact_dir
    except:
        pass

    try:
        manifest = artifact._load_manifest()
        filepath = artifact_dir / list(manifest.entries)[0]
        d["filepath"] = filepath        
    except:
        pass

    out = SimpleNamespace(**d)

    return out

def shaper_slow(df_visits, output_file=None):
    """ Convert df of time stamped vessel visits to ports to links, using a for loop over the groups of visits for each vessel

    
    """
        
    df_visits['start'] = pd.to_datetime(df_visits['start']) 
    df_visits['end'] = pd.to_datetime(df_visits['end']) 
        
    l = []
    groups = df_visits.groupby('uid')

    for name, dfg in tqdm(groups, desc="Computing edge lists", position=0, leave=True):
        for i in range(0, len(dfg)):

            if i+1 < len(dfg):            
                l.append({'start_port':dfg.iloc[i].port,
                 'start_region':dfg.iloc[i].region,
                 'start_port_name':dfg.iloc[i].port_name,
                 'start_date':dfg.iloc[i].end,
                 'end_port':dfg.iloc[i+1].port,
                 'end_region':dfg.iloc[i+1].region,
                 'end_port_name':dfg.iloc[i+1].port_name,
                 'end_date':dfg.iloc[i+1].start,
                 'duration':dfg.iloc[i+1].start - dfg.iloc[i].end,
                 'mmsi':dfg.iloc[i].mmsi,
                 'uid':dfg.iloc[i].uid,
                 ##'cargo':dfg.iloc[i].cargo,
                 'vesseltype':dfg.iloc[i].vesseltype,
                })


    df_edges = pd.DataFrame(data = l)
    
    df_edges["duration_seconds"] = df_edges["duration"].dt.total_seconds()
    df_edges.drop(columns=["duration"], inplace=True)

    if output_file is not None:
         if ".csv" in str(output_file):
            df_edges.to_csv(output_file)
         elif ".parquet" in str(output_file):
            df_edges.to_parquet(output_file)

    

    return df_edges

def set_types_edge_list(df_edges):
    df_edges["duration_days"] = df_edges["duration_seconds"]/(60*60*24)
    df_edges["vesseltype"] = df_edges["vesseltype"].astype('category')
    df_edges["start_port"] = df_edges["start_port"].astype('category')
    df_edges["end_port"] =df_edges["end_port"].astype(df_edges["start_port"].dtype)
    df_edges["start_port_name"] =df_edges["start_port_name"].astype('category')
    df_edges["end_port_name"] =df_edges["end_port_name"].astype(df_edges["start_port_name"].dtype)
    df_edges["start_region"] =df_edges["start_region"].astype('category')
    df_edges["end_region"] =df_edges["end_region"].astype(df_edges["end_region"].dtype)
    df_edges["mmsi"] = df_edges["mmsi"].astype('category')
    df_edges["uid"] = df_edges["uid"].astype('category')
    return df_edges


def save_parquet_and_wandb_log_locally(run, df, name, fold):
    
    # save locally and log it as local artifact, visible in wandb
    filepath = get_data_path() / fold / f"{name}.parquet"
    logger.info(f"Saving dataframe {name} in {filepath}")
    df.to_parquet(filepath)
    # save locally and log it as (local) artifact
    artifact = wandb.Artifact(name, type='dataset')
    artifact.add_reference('file:///' + str(filepath))
    run.log_artifact(artifact)
    logger.info(f"Logged it as artifact")

def get_latest_port_data(run=None):
    proj_name=get_project_name()
    if run is not None:

        dir = run.use_artifact(f"{proj_name}/ports_features:latest").download(root=get_wandb_root_path())
        df_feat = pd.read_parquet(Path(dir) / 'ports_features.parquet')

        dir = run.use_artifact(f"{proj_name}centralities-ports:latest").download(root=get_wandb_root_path())
        df_centr = pd.read_parquet(Path(dir) / 'centralities-ports.parquet')
    else:
        api = wandb.Api()
        dir = api.artifact(f"{proj_name}/ports_features:latest").download(root=get_wandb_root_path())
        df_feat = pd.read_parquet(Path(dir) / 'ports_features.parquet')

        dir = api.artifact(f"{proj_name}/centralities-ports:latest").download(root=get_wandb_root_path())
        df_centr = pd.read_parquet(Path(dir) / 'centralities-ports.parquet')



    return {"centralities": df_centr, "features": df_feat}

    