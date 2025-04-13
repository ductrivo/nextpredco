"""
Plotting functions for transient simulations.

This module contains functions for plotting transient simulations.

Functions:

- `plot_transient_multi_systems`: Plot transient simulations for multiple control systems.
- `plot_transient`: Plot transient simulations for a single model.
- `_plot_physical_var`: Plot a physical variable for a model.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nextpredco.core._sync import GlobalState
from nextpredco.core.control_system import ControlSystem
from nextpredco.core.model._model2 import ModelABC
from nextpredco.core.settings import GraphicsSettings

SS_VARS_ORDER = 'xyzmoupq'

settings = GraphicsSettings()


def plot_transient_multi_systems(
    systems: dict[str, ControlSystem],
    k0: int = 0,
    kf: int | None = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot transient simulations for multiple control systems.

    Parameters
    ----------
    systems : dict[str, ControlSystem]
        Dictionary with control systems to plot.
    k0 : int, optional
        Initial time step for plotting, by default 0.
    kf : int | None, optional
        Final time step for plotting, by default None.

    Returns
    -------
    fig : Figure
        Figure object.
    axs : list[Axes]
        List of axes objects.
    """

    # Get the physical variables to plot
    all_vars: dict[str, str] = {}
    for ss_var in ['m', 'u', 'p', 'q']:
        for system in systems.values():
            for name in getattr(GlobalState, f'{ss_var}_vars')():
                if name not in all_vars:
                    all_vars[name] = ss_var

    # Compute the number of axes
    n_ax = len(all_vars)

    # Create the figure and axes
    fig: Figure
    axs: list[Axes]
    fig, ax_ = plt.subplots(n_ax, 1, sharex=True, figsize=(12, 9))

    axs = ax_ if n_ax > 1 else [ax_]

    # Plot the transient simulations
    for name, system in systems.items():
        kf_ = kf if kf is not None else GlobalState.k()
        for i, (physical_var, ss_var) in enumerate(all_vars.items()):
            ax = axs[i]

            if physical_var in GlobalState.x_vars() + GlobalState.u_vars():
                t_preds = system.controller.predictions.t.arr.T
                val_preds = system.controller.predictions.get_var(
                    physical_var
                ).arr.T
                # val_preds = system.controller.predictions.get_var(
                #     physical_var
                # ).arr.T
                # fine_preds = system.controller.predictions.get_var(
                #     physical_var
                # ).arr.T

                ax.plot(
                    t_preds,
                    val_preds,
                    **settings.get_style(source='pred'),  # type: ignore[arg-type]
                )
                # ax.plot(
                #     t_preds,
                #     fine_preds,
                #     **settings.get_style(source='fine_pred'),  # type: ignore[arg-type]
                # )

            attr = system.plant.get_var(physical_var)
            t = system.plant.t.get_hist(k0, kf_)[0, :]
            val = attr.get_hist(k0, kf_)[0, :]

            prefix = '' if i != 0 else f'{name} - '.replace('.csv', '')
            ax.plot(t, val, **settings.get_style(source='est', prefix=prefix))  # type: ignore[arg-type]

            # input(f'system.plant._settings.tex = {system.plant._settings.tex}')
            ax.set_ylabel(f'${system.plant._settings.tex[physical_var]}$')

        # Set the limits and labels for the axes
        x_max_list: list[float] = []
        x_min_list: list[float] = []
    for ax in axs:
        # Compute the same limits for all axes
        x_min = min([line.get_data()[0].min() for line in ax.get_lines()])
        x_max = max([line.get_data()[0].max() for line in ax.get_lines()])
        y_min = min([line.get_data()[1].min() for line in ax.get_lines()])
        y_max = max([line.get_data()[1].max() for line in ax.get_lines()])

        x_max_list.append(x_max)
        x_min_list.append(x_min)

        # Add padding to the y limits
        if np.isclose(y_min, y_max):
            y_min = y_min * 0.9
            y_max = y_max * 1.1
        else:
            y_min = y_min - (y_max - y_min) * 0.1
            y_max = y_max + (y_max - y_min) * 0.1

        # Set the limits for all axes
        if not np.isclose(y_min, y_max):
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(bottom=y_min)

        ax.grid(linestyle=':')

    axs[0].legend(ncol=len(systems), loc='best')
    axs[-1].set_xlabel('Time [s]')
    ax.set_xlim(min(x_min_list), max(x_max_list))

    return fig, axs


def plot_transient(
    model: ModelABC,
    k0: int = 0,
    kf: int = -1,
) -> tuple[Figure, list[Axes]]:
    """
    Plot transient simulations for a single model.

    Parameters
    ----------
    model : ModelABC
        ModelABC to plot.
    k0 : int, optional
        Initial time step for plotting, by default 0.
    kf : int, optional
        Final time step for plotting, by default -1.

    Returns
    -------
    fig : Figure
        Figure object.
    ax : list[Axes]
        List of axes objects.
    """
    kf_ = GlobalState.k() if kf == -1 else kf

    # Compute the number of axes
    # TODO Remove duplicated physical vars
    n_ax = len(GlobalState.y_vars() + GlobalState.u_vars())

    fig: Figure
    ax: list[Axes]
    fig, ax_ = plt.subplots(n_ax, 1, sharex=True, figsize=(12, 9))

    ax = [ax_] if n_ax == 1 else ax_

    for i, name in enumerate(model._settings.y_vars + model._settings.u_vars):
        _plot_physical_var(
            ax=ax[i],
            model=model,
            name=name,
            k0=k0,
            kf=kf_,
            source='est',
        )

    ax[-1].set_xlabel('Time [s]')
    return fig, ax


def _plot_physical_var(
    ax: Axes,
    model: ModelABC,
    name: str,
    k0: int,
    kf: int,
    source: str,
) -> None:
    """
    Plot a physical variable for a model.

    Parameters
    ----------
    ax : Axes
        Axes object to plot on.
    model : ModelABC
        ModelABC to plot.
    name : str
        Physical variable name.
    k0 : int
        Initial time step for plotting.
    kf : int
        Final time step for plotting.
    source : str
        Source of the data to plot.
    """
    t = model.t.get_hist(0, kf)[0, :]
    val = model.get_var(name).get_hist(k0, kf)[0, :]
    ax.plot(t, val, label=f'{source}')
    # ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'${model._settings.tex[name]}$')
    ax.set_xlim(t.min(), t.max())

    ax.legend(loc='best')
    ax.grid(linestyle=':')
