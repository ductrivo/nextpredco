import contextlib
import json
import shutil
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

with contextlib.suppress(ImportError):
    from rich import print  # noqa: A004


def copy_example_data(example_data: str, destination: Path):
    data_dir = Path(__file__).parents[1] / 'data' / example_data
    shutil.copytree(src=data_dir, dst=destination)


def dict_to_str(input_: dict) -> str:
    str_ = ''

    max_key_length = len(str(max(input_.keys(), key=lambda x: len(str(x)))))

    for key, value in input_.items():
        str_ += f'\t{key:{max_key_length}}:\t{value}\n'

    return str_[:-1]


def list_to_str(input_: list) -> str:
    return '[' + ', '.join(map(str, input_)) + ']'


def print_dict(input_: dict):
    print(dict_to_str(input_))


def pretty_print_dict(data: dict):
    print(json.dumps(data, indent=4))


def is_in_list(
    t: float, t_grid: list[float | int] | NDArray, tol: float = 1e-6
) -> bool:
    return any(np.isclose(t, tg, atol=tol) for tg in t_grid)
