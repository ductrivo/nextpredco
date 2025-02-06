from pathlib import Path

from numpy.typing import NDArray

from nextpredco.core import CONFIG_FOLDER, graphics, logger, tools
from nextpredco.core.control_system import (
    ControlSystem,
    construct_control_system,
)
from nextpredco.core.examples import get_example_data


def simulate_transient(
    system: ControlSystem,
    x_arr: NDArray,
    u_arr: NDArray,
    p_arr: NDArray,
):
    for k in range(u_arr.shape[1]):
        system.model.make_step(
            x=x_arr[:, k, None],
            u=u_arr[:, k, None],
            p=p_arr[:, k, None],
        )
    return system


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


def simulation_with_example_data(
    setting_file_name: str = 'settings_template.csv',
):
    ex_data = get_example_data()
    system = construct_control_system(setting_file_name=setting_file_name)
    x_arr = ex_data['x'].T
    u_arr = ex_data['u'].T
    p_arr = ex_data['p'].T
    return simulate_transient(system, x_arr, u_arr, p_arr)


def simulation_different_settings():
    systems = {
        file_name: simulation_with_example_data(file_name)
        for file_name in ['settings_taylor.csv']
    }
    graphics.plot_transient_multi_systems(systems)
