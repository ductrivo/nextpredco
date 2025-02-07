import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nextpredco.core.control_system import ControlSystem
from nextpredco.core.model import Model
from nextpredco.core.settings import GraphicsSettings

SS_VARS_ORDER = 'xyzmoupq'

settings = GraphicsSettings()


def plot_transient_multi_systems(
    systems: dict[str, ControlSystem],
    k0: int = 0,
    kf: int | None = None,
):
    kf = (
        kf
        if kf is not None
        else max([system.model.k for system in systems.values()])
    )
    one_system = len(systems) == 1

    all_vars: dict[str, str] = {}
    for ss_var in ['y', 'u', 'p', 'q']:
        for system in systems.values():
            for name in getattr(system.model._settings, f'{ss_var}_vars'):
                if name not in all_vars:
                    all_vars[name] = ss_var

    n_ax = len(all_vars)

    fig: Figure
    axs: list[Axes]
    fig, ax_ = plt.subplots(n_ax, 1, sharex=True, figsize=(12, 9))

    axs = ax_ if n_ax > 1 else [ax_]

    for name, system in systems.items():
        for i, (physical_var, ss_var) in enumerate(all_vars.items()):
            ax = axs[i]

            attribute = system.model.get_var(physical_var)
            t = system.model.t.get_hist(k0, system.model.k)[0, :]
            val = attribute.est.get_hist(k0, system.model.k)[0, :]

            prefix = '' if i != 0 else f'{name} - '.replace('.csv', '')

            ax.plot(t, val, **settings.get_style(source='est', prefix=prefix))  # type: ignore[arg-type]

            ax.set_ylabel(f'${system.model.settings.tex[physical_var]}$')

    for ax in axs:
        # Compute the same limits for all axes
        x_min = min([line.get_data()[0].min() for line in ax.get_lines()])
        x_max = max([line.get_data()[0].max() for line in ax.get_lines()])
        y_min = min([line.get_data()[1].min() for line in ax.get_lines()])
        y_max = max([line.get_data()[1].max() for line in ax.get_lines()])

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
    ax.set_xlim(x_min, x_max)

    return fig, axs


def plot_transient(
    model: Model,
    k0: int = 0,
    kf: int = -1,
):
    kf_: int = model.k if kf == -1 else kf

    # Compute the number of axes
    # TODO Remove duplicated physical vars
    n_ax = len(model._settings.y_vars + model._settings.u_vars)

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
    model: Model,
    name: str,
    k0: int,
    kf: int,
    source: str,
) -> None:
    t = model.t.get_hist(0, kf)[0, :]
    val = model.get_var(name).est.get_hist(k0, kf)[0, :]
    ax.plot(t, val, label=f'{source}')
    # ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'${model._settings.tex[name]}$')
    ax.set_xlim(t.min(), t.max())

    ax.legend(loc='best')
    ax.grid(linestyle=':')
