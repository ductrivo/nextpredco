from collections.abc import Generator
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture
def log_dir() -> Generator[Path, None, None]:
    # Set the log directory for the tests
    log_dir = PROJECT_DIR / 'logs_test'

    yield log_dir
    print('Finished')
    # # Remove folder after the test
    # if log_dir.exists():
    #     shutil.rmtree(log_dir)
