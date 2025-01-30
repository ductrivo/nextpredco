from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'

SS_VARS_DB = ['x', 'z', 'upq']
SS_VARS_PRIMARY = ['x', 'z', 'u', 'p', 'q']
SS_VARS_SECONDARY = ['m', 'o', 'y']

SS_VARS_SOURCES = ['goal', 'est', 'act', 'meas', 'filt']


CONFIG_FOLDER = '_configs'
SETTING_FOLDER = '_settings'
