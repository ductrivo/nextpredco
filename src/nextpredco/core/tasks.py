from pathlib import Path

from nextpredco.core import (
    CONFIG_FOLDER,
    logger,
    tools,
)
from nextpredco.core.control_system import construct_control_system
from nextpredco.core.examples import get_example_data
from nextpredco.core.graphics import plot_transient
from nextpredco.core.model import Model


def simulate_transient():
    model = Model()

    for _ in range(model.n_max):
        model.make_step()

    plot_transient(model)


def init_dir(work_dir: Path, example_project: str):
    try:
        config_dir = work_dir / CONFIG_FOLDER
        tools.copy_example_data(
            example_data=example_project,
            destination=config_dir,
        )
    except FileExistsError:
        logger.error(
            'Directory %s already exists. Please remove remove it first.',
            CONFIG_FOLDER,
        )


def compare_with_do_mpc():
    ex_data = get_example_data()
    system = construct_control_system()
    x_arr = ex_data['x'].T
    u_arr = ex_data['u'].T
    p_arr = ex_data['p'].T
    for k in range(u_arr.shape[1]):
        system.model.make_step(
            x=x_arr[:, k, None],
            u=u_arr[:, k, None],
            p=p_arr[:, k, None],
        )
    plot_transient(system.model)
