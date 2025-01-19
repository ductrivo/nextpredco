"""
This file is used to add the parent directory to the sys.path
so that the tests can import the modules from the parent directory.
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
THIS_DIR_PARENT = (THIS_DIR / '..').resolve()

sys.path.insert(0, str(THIS_DIR_PARENT))

pytest_plugins = [
    'tests.fixtures.log_dir',
]
