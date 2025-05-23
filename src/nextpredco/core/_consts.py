from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'

SS_VARS_DB = ['x', 'z', 'upq']
SS_VARS_PRIMARY = ['x', 'z', 'u', 'p', 'q']
SS_VARS_SECONDARY = ['m', 'y']

SS_VARS_SOURCES = ['goal', 'est', 'act', 'meas', 'filt']


CONFIG_FOLDER = '_configs'
SETTING_FOLDER = '_settings'

PARAMETER = 'parameter'
TYPE = 'type'
VALUE = 'value'
TEX = 'tex'
DESCRIPTION = 'description'
ROLE = 'role'
COST_ELEMENTS = ['x', 'y', 'u', 'du', 'total']
PREDICTION_ELEMENTS = ['k', 't', 'x', 'z', 'u']
