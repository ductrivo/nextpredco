import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nextpredco.core.model import Model

SS_VARS_ORDER = 'xyzmoupq'


def plot_transient(
    model: Model,
    k0: int = 0,
    kf: int = -1,
) -> None:
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
    plt.show()


def _plot_physical_var(
    ax: Axes,
    model: Model,
    name: str,
    k0: int,
    kf: int,
    source: str,
) -> None:
    t = model.t.get_hist(0, kf)[0, :]
    val_ = model.get_var(name).est.get_hist(k0, kf)[0, :]
    ax.plot(t, val_, label=f'{source}')
    # ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'${model._settings.tex[name]}$')
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(val_.min(), val_.max())
    ax.legend()
    ax.grid(linestyle=':')
