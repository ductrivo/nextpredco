from pathlib import Path

import casadi as ca
from casadi import SX as Symbolic

from nextpredco.core._logger import logger as logger

type SourceType = list[str] | tuple[str, ...]


PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'

SS_VARS_DB = ['x', 'z', 'upq']
SS_VARS_PRIMARY = ['x', 'z', 'u', 'p', 'q']
SS_VARS_SECONDARY = ['m', 'o', 'y']

SS_VARS_SOURCES = ['goal', 'est', 'act', 'meas', 'filt']


CONFIG_FOLDER = '_configs'
SETTING_FOLDER = '_settings'

PARAMETER = 'parameter'
TYPE = 'type'
VALUE = 'value'
TEX = 'tex'
DESCRIPTION = 'description'
ROLE = 'role'
