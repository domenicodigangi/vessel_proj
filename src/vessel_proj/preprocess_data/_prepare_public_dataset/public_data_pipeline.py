from prefect import flow, task
import pandas as pd
from pathlib import Path
from vessel_proj.preprocess_data._prepare_public_dataset import aggregated_graph_from_edge_list

import logging
from prefect import task
from vessel_proj.preprocess_data import get_data_path

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

# THIS PIPELINE HAS NEVER BEEN TESTED AND IT ASSUMES THAT THE RAW VISITS' LIST IS ALREADY IN DATA/RAW


@task
def clean_visits(dir: Path, min_dur_sec=300, types_to_drop = ["fishing", "tug tow"]):
        
        df = pd.read_parquet(dir / 'visits-augmentend.parquet')

        inds = df["duration_seconds"] < min_dur_sec
        logger.info(f"Dropping {inds.sum()} rows for too low duration")
        df.drop(df[inds].index, inplace=True)

        
        inds = df["vessel_category"].apply(lambda x: x in types_to_drop)
        logger.info(f"Dropping {inds.sum()} rows because vessel type in {types_to_drop}")
        df.drop(df[inds].index, inplace=True)


        # store 
        file_path = get_data_path() / "interim" / 'visits-augmented-cleaned.parquet'
        df.to_parquet(file_path)



@task
def create_and_log_edge_list_from_visits(nrows=None):

        dir = get_data_path() / "raw" 

        input_file = Path(dir) / 'visits-augmented-cleaned.parquet'

        output_file = get_data_path() / "interim" /'edge_list.parquet'

        df_visits = pd.read_parquet(input_file)

        if nrows is not None:
            df_visits = df_visits[:nrows]            
            output_file = Path(output_file)
            no_suff_name = output_file.name.replace(output_file.suffix, '')
            output_file = Path(output_file.parent) / ( f"{no_suff_name}_from_{nrows}_visits{output_file.suffix}"  )


        df_edges = shaper_slow(df_visits, output_file=output_file)

  

@flow
def main():
    dir = get_data_path() / "raw" 
    df_visits = pd.read_csv(dir / 'visits-augmented.csv')
    df_visits.to_parquet(dir / 'visits-augmented.parquet')
    
    clean_visits(dir)

    create_and_log_edge_list_from_visits()

    aggregated_graph_from_edge_list.main()
    
if __name__ == '__main__':
    main()