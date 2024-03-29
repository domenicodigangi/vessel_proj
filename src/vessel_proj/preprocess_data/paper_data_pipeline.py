from vessel_proj.preprocess_data import get_data_path
from vessel_proj.preprocess_data.get_clean_and_save_wpi_feat import (
    get_clean_and_save_wpi_feat,
)
from vessel_proj.preprocess_data.get_and_save_centralities_from_graph_data import (
    get_graph_data,
    get_and_save_centralities_from_graph_data,
)
from vessel_proj.preprocess_data._prepare_public_dataset.public_data_pipeline import (
    main as process_visits,
)


def data_pipeline():
    """
    Starting from zenodo data, compute centralities and load wpi
    """

    load_path = get_data_path() / "interim"
    save_path = get_data_path() / "processed"

    try:
        get_graph_data()
    except FileNotFoundError:
        print("Trying to clean data from visits list, will take a bit of time")
        process_visits()

    df_wpi_port_feat = get_clean_and_save_wpi_feat(load_path, save_path)

    ports_to_exclude_list = [
        # "EUROPA POINT",
        # "CEUTA",
        # "VACAMONTE",
        # "KATSUNAN KO",
        # "TONGUE POINT",
    ]
    ports_to_exclude = {
        row["PORT_NAME"]: id
        for id, row in df_wpi_port_feat.iterrows()
        if row["PORT_NAME"] in ports_to_exclude_list
    }

    get_and_save_centralities_from_graph_data(save_path, ports_to_exclude)


if __name__ == "__main__":
    data_pipeline()
