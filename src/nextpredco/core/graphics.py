import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nextpredco.core.model import Model

SS_VARS_ORDER = 'xyzmoupq'


def plot_transient(
    model: Model,
    kf: int = -1,
    ss_vars: str = 'yup',
    plot_vars: list[str] | str = 'yup',
) -> None:
    kf_: int = model.k if kf == -1 else kf
    t0 = model.t.get_val(0)[0, 0]
    tf = model.t.get_val(kf_)[0, 0]

    # Compute the number of axes
    # TODO Remove duplicated physical vars
    n_ax = len(plot_vars)

    fig: Figure
    ax: list[Axes]
    fig, ax_ = plt.subplots(n_ax, 1, sharex=True, figsize=(12, 9))

    ax = [ax_] if n_ax == 1 else ax_

    t = model.t.get_hist(0, model.k)[0, :]
    x = model.get_var('c_a').est.get_hist(0, model.k)[0, :]

    ax[0].plot(t, x, label='x_est')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel(f'${model._settings.tex["c_a"]}$')
    ax[0].set_xlim(t0, tf)
    ax[0].set_ylim(x.min(), x.max())
    plt.show()
