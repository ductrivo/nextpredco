import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nextpredco.core.consts import CONFIG_FOLDER

DATA_DIR = Path(__file__).parent


def list_example_projects() -> list[str]:
    # List all folders in data_dir

    return [
        f.name for f in DATA_DIR.iterdir() if f.is_dir() and '__' not in f.name
    ]


def get_example_data() -> dict[str, NDArray]:
    data_path = Path.cwd() / CONFIG_FOLDER / 'transient_data.npz'

    with np.load(data_path) as data:
        return dict(data)


if __name__ == '__main__':
    print(get_example_data())
