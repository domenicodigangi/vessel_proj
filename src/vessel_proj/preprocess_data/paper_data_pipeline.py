from prefect import flow
from . import get_and_clean_wpi_feat
from . import get_centralities_from_graph_data


@flow
def run():

    get_centralities_from_graph_data.main()
    get_and_clean_wpi_feat.main()
    

