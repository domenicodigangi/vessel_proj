import functools

from pathlib import Path
import logging
import pandas as pd
from tqdm import tqdm
import logging
import wandb
from pathlib import Path
from types import SimpleNamespace
from prefect import task
from dotenv import load_dotenv
import os


logger = logging.getLogger()
logger.setLevel(logging.INFO)


load_dotenv()
_proj_name = os.environ.get('wandb_proj_name')
_wandb_entity = os.environ.get('wandb_entity')


def create_data_fold_structure(project_dir):
    logger.info(f"Creating data folder structure in {project_dir}")
    Path.mkdir(project_dir / "data", exist_ok=True)
    Path.mkdir(project_dir / "data" / "raw", exist_ok=True)
    Path.mkdir(project_dir / "data" / "processed", exist_ok=True)
    Path.mkdir(project_dir / "data" / "interim", exist_ok=True)
    Path.mkdir(project_dir / "data" / "external", exist_ok=True)


def get_project_name():
    return _proj_name


def get_wandb_entity():
    return _wandb_entity


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


def catch_all_and_log(f, logger=None):
    """
    A function wrapper for catching all exceptions and logging them
    """
    @functools.wraps(f)
    def inner(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        try:
            return f(*args, **kwargs)
        except Exception as ex:
            if logger is None:
                print(ex)
            else:
                logger.error(ex)

    return inner
