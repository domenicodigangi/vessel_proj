from pathlib import Path
import logging 


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent


def get_raw_data_path():
    logging.warning(f"using hardcoded path for raw data in barbera" )
    proj_root = get_project_root() 
    raw_data_path = proj_root.parent.parent.parent / "data" / "digiandomenico" / "vessel_proj"

    logging.warning(f"using hardcoded path for raw data in barbera: {raw_data_path}" )

    return raw_data_path