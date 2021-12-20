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
    root_path.mkdir(exist_ok=True, parents=True)
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

def shaper_slow(input_file, output_file, nrows=None):
    """ Convert df of time stamped vessel visits to ports to links, using a for loop over the groups of visits for each vessel

    
    """
    
    df_compact = pd.read_csv(input_file, nrows=nrows)

    df_compact['start'] = pd.to_datetime(df_compact['start']) 
    df_compact['end'] = pd.to_datetime(df_compact['end']) 
    
    
    l = []
    groups = df_compact.groupby('uid')

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
    if nrows is not None:
        output_file = Path(output_file)
        no_suff_name = output_file.name.replace(output_file.suffix, '')
        output_file = Path(output_file.parent) / ( f"{no_suff_name}_from_{nrows}_visits{output_file.suffix}"  )


    df_edges.to_csv(output_file)

    return df_edges

def read_edge_list(filepath):
    df_edges = pd.read_csv(filepath)
    df_edges["start_date"] = pd.to_datetime(df_edges["start_date"])
    df_edges["end_date"] = pd.to_datetime(df_edges["end_date"])
    df_edges["duration"] = pd.to_timedelta(df_edges["duration"])
    df_edges["duration_days"] = df_edges["duration"].dt.total_seconds()/(60*60*24)
    df_edges["vesseltype"] = df_edges["vesseltype"].astype('category')
    df_edges["start_port"] = df_edges["start_port"].astype('category')
    df_edges["end_port"] =df_edges["end_port"].astype(df_edges["start_port"].dtype)
    df_edges["start_port_name"] =df_edges["start_port_name"].astype('category')
    df_edges["end_port_name"] =df_edges["end_port_name"].astype(df_edges["start_port_name"].dtype)
    df_edges["start_region"] =df_edges["start_region"].astype('category')
    df_edges["end_region"] =df_edges["end_region"].astype(df_edges["end_region"].dtype)
    df_edges["mmsi"] =df_edges["mmsi"].astype('category')
    df_edges["uid"] =df_edges["uid"].astype('category')
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


    