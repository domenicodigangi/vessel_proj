from pathlib import Path
import logging 
import pandas as pd
from tqdm import tqdm 
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)



def create_data_fold_structure(project_dir):
    logger.info(f"Creating data folder structure in {project_dir}")
    Path.mkdir(project_dir / "data", exist_ok = True)
    Path.mkdir(project_dir / "data" / "raw", exist_ok = True)
    Path.mkdir(project_dir / "data" / "processed", exist_ok = True)
    Path.mkdir(project_dir / "data" / "interim", exist_ok = True)
    Path.mkdir(project_dir / "data" / "external", exist_ok = True)




def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent


def get_data_path():
    proj_root = get_project_root() 
    data_path = proj_root.parent.parent.parent / "data" / "digiandomenico" / "vessel_proj" / "data"

    logging.warning(f"using hardcoded path for data in barbera: {data_path}" )

    return data_path



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


