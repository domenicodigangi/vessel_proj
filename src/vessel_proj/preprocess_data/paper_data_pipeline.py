from prefect import flow
from . import get_data_path
from . import get_and_clean_wpi_feat
from . import get_centralities_from_graph_data


@flow
def run():

    load_path = get_data_path() / "interim"
    save_path = get_data_path() / "processed"

    get_centralities_from_graph_data.main(load_path, save_path)
    get_and_clean_wpi_feat.main(load_path, save_path)
    

