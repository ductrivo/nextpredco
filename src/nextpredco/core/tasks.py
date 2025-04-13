from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from nextpredco.core import graphics, tools
from nextpredco.core._consts import CONFIG_FOLDER
from nextpredco.core._logger import logger
from nextpredco.core.control_system import (
    construct_control_system,
)
from nextpredco.core.examples import get_example_data


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


def simulation_different_settings():
    ex_data = get_example_data()
    x_arr = ex_data['x'].T
    u_arr = ex_data['u'].T
    p_arr = ex_data['p'].T
    t_grid = np.arange(0, x_arr.shape[1] + 1) * 0.01

    systems = {
        file_name: simulation_with_example_data(file_name)
        for file_name in ['settings_taylor.csv', 'settings_idas.csv']
    }
    fig, axs = graphics.plot_transient_multi_systems(systems)
    axs[0].plot(t_grid[:-1], x_arr[0, :], label='ground truth')
    axs[1].plot(t_grid[:-1], x_arr[1, :], label='ground truth')
    plt.show()


def simulation_with_example_data(
    setting_file_name: str = 'settings_template.csv',
):
    system = construct_control_system(setting_file_name=setting_file_name)

    ex_data = get_example_data()
    x_arr = ex_data['x'].T
    u_arr = ex_data['u'].T
    p_arr = ex_data['p'].T
    t_grid = np.arange(0, x_arr.shape[1] + 1) * 0.01

    system.simulate_model_only(
        t_grid=t_grid,
        x_arr=x_arr,
        u_arr=u_arr,
        p_arr=p_arr,
    )
    # input(x_arr)
    return system


def simulate_control_system():
    system = construct_control_system('settings_template.csv')
    system.simulate()
    graphics.plot_transient_multi_systems({'taylor': system})
    plt.show()
