from prefect import flow, task
import pandas as pd
from . import get_and_clean_wpi_feat
from . import get_centralities_from_graph_data


@flow
def main():

    get_centralities_from_graph_data.main()
    get_and_clean_wpi_feat.main()
    
