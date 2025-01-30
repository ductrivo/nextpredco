import ast
import contextlib
import itertools
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from types import UnionType

import pandas as pd

from nextpredco.core import utils
from nextpredco.core.consts import (
    CONFIG_FOLDER,
    SETTING_FOLDER,
    SS_VARS_PRIMARY,
    SS_VARS_SECONDARY,
)
from nextpredco.core.control_system import ControlSystem
from nextpredco.core.errors import SystemVariableError
from nextpredco.core.graphics import plot_transient
from nextpredco.core.logger import logger
from nextpredco.core.model import Model, ModelSettings


def simulate_transient():
    model = Model()

    for _ in range(model.n_max):
        model.make_step()

    plot_transient(model)


def init_dir(work_dir: Path, example_project: str):
    try:
        config_dir = work_dir / CONFIG_FOLDER
        # config_dir.mkdir(parents=True)
        utils.copy_example_data(
            example_data=example_project,
            destination=config_dir,
        )
    except FileExistsError:
        logger.error(
            'Directory %s already exists. Please remove remove it first.',
            CONFIG_FOLDER,
        )
