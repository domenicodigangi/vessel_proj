from prefect import flow
from vessel_proj.preprocess_data import get_data_path
from vessel_proj.preprocess_data.get_clean_and_save_wpi_feat  import get_clean_and_save_wpi_feat
from vessel_proj.preprocess_data.get_and_save_centralities_from_graph_data  import get_and_save_centralities_from_graph_data



@flow
def data_pipeline():
    """
    Starting from zenodo data, compute centralities and load wpi
    """

    load_path = get_data_path() / "interim"
    save_path = get_data_path() / "processed"

    get_and_save_centralities_from_graph_data(save_path)
    get_clean_and_save_wpi_feat(load_path, save_path)
    

if __name__ == "__main__":
    data_pipeline()