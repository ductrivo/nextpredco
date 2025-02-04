import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import NDArray
from rich.pretty import pretty_repr

from nextpredco.core import (
    CONFIG_FOLDER,
    SS_VARS_DB,
    Symbolic,
    logger,
)
from nextpredco.core.integrator import IDAS
from nextpredco.core.model._descriptors import (
    PhysicalVariable,
    ReadOnlyFloat,
    ReadOnlyInt,
    ReadOnlySource,
    SystemVariable,
    TimeVariable,
    VariableSource,
)
from nextpredco.core.settings._settings import ModelSettings

try:
    from rich.pretty import pretty_repr
except ImportError:

    def pretty_repr(obj: Any) -> Any:  # type: ignore[misc]
        return obj


from collections.abc import Callable

from numpy.typing import ArrayLike


@dataclass
class ModelData:
    k: int
    k_clock: int
    n_max: int
    n_clock_max: int

    t_full: NDArray
    t_clock_full: NDArray
    x_est_full: NDArray
    z_est_full: NDArray
    upq_est_full: NDArray

    x_goal_full: NDArray | None = field(default=None)
    z_goal_full: NDArray | None = field(default=None)
    upq_goal_full: NDArray | None = field(default=None)

    x_act_full: NDArray | None = field(default=None)
    z_act_full: NDArray | None = field(default=None)
    upq_act_full: NDArray | None = field(default=None)

    x_meas_full: NDArray | None = field(default=None)
    z_meas_full: NDArray | None = field(default=None)
    upq_meas_full: NDArray | None = field(default=None)

    x_filt_full: NDArray | None = field(default=None)
    z_filt_full: NDArray | None = field(default=None)
    upq_filt_full: NDArray | None = field(default=None)

    x_goal_clock_full: NDArray | None = field(default=None)
    z_goal_clock_full: NDArray | None = field(default=None)
    upq_goal_clock_full: NDArray | None = field(default=None)

    x_act_clock_full: NDArray | None = field(default=None)
    z_act_clock_full: NDArray | None = field(default=None)
    upq_act_clock_full: NDArray | None = field(default=None)

    x_est_clock_full: NDArray | None = field(default=None)
    z_est_clock_full: NDArray | None = field(default=None)
    upq_est_clock_full: NDArray | None = field(default=None)

    x_meas_clock_full: NDArray | None = field(default=None)
    z_meas_clock_full: NDArray | None = field(default=None)
    upq_meas_clock_full: NDArray | None = field(default=None)

    x_filt_clock_full: NDArray | None = field(default=None)
    z_filt_clock_full: NDArray | None = field(default=None)
    upq_filt_clock_full: NDArray | None = field(default=None)


class Model:
    # Settings
    dt = ReadOnlyFloat()
    t_max = ReadOnlyFloat()

    k_clock = ReadOnlyInt()
    dt_clock = ReadOnlyFloat()
    sources = ReadOnlySource()

    # Data
    n_clock_max = ReadOnlyInt()
    n_max = ReadOnlyInt()

    x = SystemVariable()
    z = SystemVariable()
    u = SystemVariable()
    p = SystemVariable()
    q = SystemVariable()
    y = SystemVariable()
    m = SystemVariable()
    o = SystemVariable()
    _physical_var = PhysicalVariable()
    upq = SystemVariable()
    t = TimeVariable()
    t_clock = TimeVariable()

    def __init__(
        self,
        # model_data_path: Path = DATA_DIR / "ex_chen1998.csv",
        settings: ModelSettings | None = None,
        integrator: IDAS | None = None,
    ) -> None:
        # Load settings
        self._settings = settings
        logger.debug('Model settings:\n%s', pretty_repr(self._settings))

        # Load integrator
        self._integrator = integrator

        # Load data
        self._data = self._create_data()

        # Load equations
        # TODO: move to data
        self._transient_eqs, self._transient_funcs = self._create_equations()

    def _create_data(self) -> ModelData:
        # Create data holder
        data: dict[str, int | NDArray] = {}

        # Extract time information
        data['k'] = 0
        data['k_clock'] = 0
        data['n_max'] = round(self._settings.t_max / self._settings.dt)
        data['n_clock_max'] = round(
            self._settings.t_max / self._settings.dt_clock,
        )

        # Create data holder (full arrays) for each source
        created_attrs: list[str] = []
        for ss_var in SS_VARS_DB:
            vars_list = getattr(self._settings, f'{ss_var}_vars')
            length = len(vars_list)
            # TODO: May have different initial values for each source

            if length > 0:
                init_val = np.array(
                    [[self._settings.info[key] for key in vars_list]],
                ).T
            else:
                init_val = np.zeros((length, 1))

            # Create data holder (full arrays) for each source
            for source in self.sources:
                n_max = (
                    data['n_clock_max'] if 'clock' in source else data['n_max']
                )
                arr_full = np.zeros((init_val.shape[0], n_max + 1))

                arr_full[:, 0, None] += init_val
                attr_name = f'_{ss_var}_{source}_full'

                data[attr_name[1:]] = arr_full
                created_attrs.append(attr_name)

        # Create data holder (full arrays) for time variables
        data['t_full'] = np.zeros((1, data['n_max'] + 1))
        data['t_clock_full'] = np.zeros((1, data['n_clock_max'] + 1))
        return ModelData(**data)  # type: ignore[arg-type]

    @staticmethod
    def _load_equations() -> tuple[
        Callable[..., Symbolic],
        Callable[..., Symbolic],
    ]:
        module_path = Path().cwd() / CONFIG_FOLDER / 'equations.py'
        spec = importlib.util.spec_from_file_location(
            'dynamic_module',
            module_path,
        )
        if spec is None:
            msg = f'Cannot load module from {module_path}'
            raise ImportError(msg)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.create_f, module.create_g

    def _create_upq_dict(
        self,
        upq: NDArray | Symbolic,
    ) -> dict[str, NDArray | Symbolic]:
        # TODO: Review the type hints
        upq_dict = {}

        for var_ in self._settings.upq_vars:
            idx = self._settings.upq_vars.index(var_)
            upq_dict[var_] = upq[idx, 0]

        return upq_dict

    def _create_equations(
        self,
    ) -> tuple[dict[str, Symbolic], dict[str, ca.Function]]:
        # Create symbolic variables
        x = Symbolic.sym('__x', self.n('x'))
        z = Symbolic.sym('__z', self.n('z'))
        upq = Symbolic.sym('__upq', self.n('upq'))

        # Create dictionary with all symbolic variables
        all_vars = self.get_all_vars(x=x, z=z, upq=upq)

        # Load equations from equations.py
        create_f, create_g = self._load_equations()
        f = create_f(**all_vars)
        f_func = ca.Function('private_func_f', [x, z, upq], [f])

        transient_eqs = {
            'x': x,
            'z': z,
            'p': upq,
            'ode': f,
        }

        transient_funcs = {
            'f': f_func,
        }

        if self.n('z') > 0:
            g = create_g(**all_vars)
            g_func = ca.Function('private_func_g', [x, z, upq], [g])
            transient_eqs['alg'] = g
            transient_funcs['f'] = g_func

        return transient_eqs, transient_funcs

    def get_y(self, x: NDArray | Symbolic):
        y: ArrayLike = []
        for var_ in self._settings.y_vars:
            idx = self._settings.y_vars.index(var_)
            y.append(x[idx, 0])

        if isinstance(x, Symbolic):
            return ca.vcat(y)

        return np.array([[y]]).T

    def get_du(self, u_arr: NDArray | Symbolic, u_last: NDArray | Symbolic):
        du: ArrayLike = [u_arr[:, 0] - u_last]
        for i in range(u_arr.shape[1] - 1):
            du.append(u_arr[:, i + 1] - u_arr[:, i])

        if isinstance(u_arr, Symbolic):
            return ca.vcat(du)

        return np.array([du]).T

    def get_all_vars(
        self, **ss_vars: Symbolic | NDArray
    ) -> dict[str, Symbolic | NDArray]:
        all_vars: dict[str, Symbolic | NDArray] = {}

        for ss_var in ['x', 'z', 'u', 'p', 'q', 'upq', 'const']:
            vars_list: list[str] = getattr(self._settings, f'{ss_var}_vars')
            for var_ in vars_list:
                if ss_var == 'const':
                    all_vars[var_] = self._settings.info[var_]
                elif ss_var in ss_vars:
                    arr_ = ss_vars[ss_var]
                    idx = vars_list.index(var_)
                    all_vars[var_] = arr_[idx, 0]
                else:
                    arr_ = getattr(self, ss_var).est.val
                    idx = vars_list.index(var_)
                    all_vars[var_] = arr_[idx, 0]
        return all_vars

    def get_upq(
        self,
        u: Symbolic | NDArray | None = None,
        p: Symbolic | NDArray | None = None,
        q: Symbolic | NDArray | None = None,
    ) -> Symbolic | NDArray:
        arr_ = []
        u = self.u.est.val if u is None else u
        p = self.p.est.val if p is None else p
        q = self.q.est.val if q is None else q

        for var_ in self._settings.u_vars:
            idx = self._settings.u_vars.index(var_)
            arr_.append(u[idx, 0])

        for var_ in self._settings.p_vars:
            idx = self._settings.p_vars.index(var_)
            arr_.append(p[idx, 0])

        for var_ in self._settings.q_vars:
            idx = self._settings.q_vars.index(var_)
            arr_.append(q[idx, 0])

        if (
            isinstance(u, Symbolic)
            or isinstance(p, Symbolic)
            or isinstance(q, Symbolic)
        ):
            return ca.vcat(arr_)

        return np.array([arr_]).T

    def _update_k_and_t(self) -> None:
        self._data.k += 1

        # TODO: try other ways to update time
        # Try case to work with t_clock
        self.t.set_val(k=self._data.k, val=self._data.k * self._settings.dt)

    @property
    def k(self) -> int:
        return self._data.k

    def get_var(self, var_: str) -> VariableSource:
        return self._physical_var(var_)

    def n(self, ss_var: str) -> int:
        return len(getattr(self._settings, f'{ss_var}_vars'))

    def make_step(
        self,
        x: NDArray | None = None,
        u: NDArray | None = None,
        p: NDArray | None = None,
        q: NDArray | None = None,
    ) -> None:
        self._update_k_and_t()

        self.u.est.val = u if u is not None else self.u.est.last
        self.p.est.val = p if p is not None else self.p.est.last
        self.q.est.val = q if q is not None else self.q.est.last

        x, z, _, _ = self._integrator.integrate(
            equations=self._transient_eqs,
            x0=self.x.est.last,
            z0=self.z.est.last,
            t_grid=self.t.get_hist(self.k - 1, self.k)[0, :],
            p0=self.upq.est.val,
        )

        self.x.est.val = x
        self.z.est.val = z

    def compute_xz(
        self,
        x0: NDArray,
        z0: NDArray,
        upq: NDArray,
        t_grid: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        x, z, x_arr, z_arr = self._integrator.integrate(
            equations=self._transient_eqs,
            x0=x0,
            z0=z0,
            p0=upq,
            t_grid=t_grid,
        )
        return x, z, x_arr, z_arr


if __name__ == '__main__':
    pass
