import contextlib
import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import numpy as np

np.set_printoptions(precision=3, suppress=True)

# from nextpredco.core.consts import PROJECT_DIR

with contextlib.suppress(ImportError):
    from rich.logging import RichHandler

    rich_handler_imported = True
    from rich.pretty import install

    install()

# Define the log directory
LOG_DIR = Path.cwd() / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Define the log file paths
LOG_FILE = LOG_DIR / 'application.log'
DEBUG_LOG_FILE = LOG_DIR / 'application_debug.log'
ERROR_LOG_FILE = LOG_DIR / 'error.log'

# Create a logger
logger = logging.getLogger('nextpredco')
logger.setLevel(logging.DEBUG)  # Set the log level

# Create formatters
simple_formatter = logging.Formatter('%(name)s:- %(message)s')
standard_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(module)s'
    ' - [in %(pathname)s:%(lineno)d]: %(message)s',
)

# Create handlers
console_handler: RichHandler | logging.StreamHandler
if rich_handler_imported:
    console_handler = RichHandler(show_path=True, rich_tracebacks=True)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(simple_formatter)
else:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(detailed_formatter)

# Create file handlers
file_handler = logging.FileHandler(
    filename=LOG_FILE,
    mode='a',
    encoding='utf-8',
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(detailed_formatter)

# Create a timed rotating file handler
file_debug_handler = TimedRotatingFileHandler(
    filename=DEBUG_LOG_FILE,
    when='W0',
    interval=1,
    backupCount=7,
    encoding='utf-8',
)
file_debug_handler.setLevel(logging.DEBUG)
file_debug_handler.setFormatter(standard_formatter)

# Create an error file handler
error_file_handler = logging.FileHandler(
    filename=ERROR_LOG_FILE,
    mode='a',
    encoding='utf-8',
)
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(detailed_formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.addHandler(file_debug_handler)
logger.addHandler(error_file_handler)

# Avoid log message propagation to the root logger
logger.propagate = False
