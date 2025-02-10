import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from rich.pretty import pretty_repr

from nextpredco.core._consts import CONFIG_FOLDER, COST_ELEMENTS, SS_VARS_DB
from nextpredco.core._data import ModelData
from nextpredco.core._logger import logger
from nextpredco.core._typing import (
    Array2D,
    ArrayType,
    FloatType,
    IntType,
    PredDType,
    SourceType,
    Symbolic,
    TgridType,
)
from nextpredco.core.integrator import IntegratorFactory, IntegratorSettings
from nextpredco.core.model._descriptors import (
    ReadOnly2,
    StateSpaceStructure,
    SystemCallable,
    SystemVariable,
)
from nextpredco.core.settings import ModelSettings


class Model:
    # Settings
    dt = ReadOnly2[FloatType]('settings')
    t_max = ReadOnly2[FloatType]('settings')

    k_clock = ReadOnly2[IntType]('settings')
    dt_clock = ReadOnly2[FloatType]('settings')
    sources = ReadOnly2[SourceType]('settings')

    x_vars = ReadOnly2[list[str]]('settings')
    z_vars = ReadOnly2[list[str]]('settings')
    u_vars = ReadOnly2[list[str]]('settings')
    p_vars = ReadOnly2[list[str]]('settings')
    q_vars = ReadOnly2[list[str]]('settings')
    y_vars = ReadOnly2[list[str]]('settings')
    m_vars = ReadOnly2[list[str]]('settings')
    o_vars = ReadOnly2[list[str]]('settings')
    upq_vars = ReadOnly2[list[str]]('settings')

    k = ReadOnly2[IntType]('data')
    k_max = ReadOnly2[IntType]('data')
    k_clock_max = ReadOnly2[IntType]('data')

    x = SystemVariable()
    z = SystemVariable()
    u = SystemVariable()
    p = SystemVariable()
    q = SystemVariable()
    y = SystemVariable()
    m = SystemVariable()
    o = SystemVariable()
    upq = SystemVariable()

    t = SystemVariable()
    t_clock = SystemVariable()
    predictions = SystemVariable()

    _physical_var = SystemCallable()

    def __init__(
        self,
        # model_data_path: Path = DATA_DIR / "ex_chen1998.csv",
        settings: ModelSettings | None = None,
        integrator_settings: IntegratorSettings | None = None,
    ) -> None:
        # Load settings
        self._settings = settings
        logger.debug('Model settings:\n%s', pretty_repr(self._settings))

        # Load data
        self._data = self._create_data()

        # Load equations
        self._equations = self._create_equations()

        # Load integrator
        if integrator_settings is not None:
            self._integrator = IntegratorFactory.create(
                settings=integrator_settings,
                equations=self._equations,
                h=self._settings.dt,
            )
        else:
            self._integrator = None

    def _create_data(self) -> ModelData:
        # Create data holder
        data: dict[str, IntType | Array2D] = {}

        # Extract time information
        data['k'] = 0
        data['k_clock'] = 0
        data['k_max'] = round(self._settings.t_max / self._settings.dt)
        data['k_clock_max'] = round(
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
                if source == 'preds':
                    continue

                k_max = (
                    data['k_clock_max'] if 'clock' in source else data['k_max']
                )
                arr_full = np.zeros((init_val.shape[0], k_max + 1))
                arr_full[:, 0, None] += init_val
                attr_name = f'_{ss_var}_{source}_full'

                data[attr_name[1:]] = arr_full
                created_attrs.append(attr_name)

        # Create data holder (full arrays) for time variables
        data['t_full'] = np.zeros((1, data['k_max'] + 1))
        data['t_clock_full'] = np.zeros((1, data['k_clock_max'] + 1))
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
        upq: ArrayType,
    ) -> dict[str, ArrayType]:
        # TODO: Review the type hints
        upq_dict = {}

        for var_ in self._settings.upq_vars:
            idx = self._settings.upq_vars.index(var_)
            upq_dict[var_] = upq[idx, 0]

        return upq_dict

    def _create_equations(
        self,
    ) -> dict[str, Symbolic]:
        # Create symbolic variables
        x = Symbolic.sym('__x', self.n('x'))
        z = Symbolic.sym('__z', self.n('z'))
        upq = Symbolic.sym('__upq', self.n('upq'))

        # Create dictionary with all symbolic variables
        all_vars = self.get_all_vars(x=x, z=z, upq=upq)

        # Load equations from equations.py
        create_f, create_g = self._load_equations()
        f = create_f(**all_vars)
        # f_func = ca.Function('private_func_f', [x, z, upq], [f])

        transient_eqs = {
            'x': x,
            'z': z,
            'p': upq,
            'ode': f,
        }

        # transient_funcs = {
        #     'f': f_func,
        # }

        if self.n('z') > 0:
            g = create_g(**all_vars)
            g_func = ca.Function('private_func_g', [x, z, upq], [g])
            transient_eqs['alg'] = g
            # transient_funcs['f'] = g_func

        return transient_eqs

    def get_y(self, x: ArrayType) -> ArrayType:
        y: ArrayLike = []

        for var_ in self._settings.y_vars:
            idx = self._settings.x_vars.index(var_)
            if isinstance(x, Symbolic):
                y.append(x[idx, :])
            else:
                y.append(x[idx, 0])

        if isinstance(x, Symbolic):
            return ca.vcat(y)

        return np.array([[y]]).T

    def get_du(self, u_arr: ArrayType, u_last: ArrayType) -> ArrayType:
        du: ArrayLike = [u_arr[:, 0] - u_last]
        for i in range(u_arr.shape[1] - 1):
            du.append(u_arr[:, i + 1] - u_arr[:, i])

        if isinstance(u_arr, Symbolic):
            return ca.hcat(du)

        return np.array([du])

    def get_all_vars(
        self, **ss_vars: ArrayType
    ) -> dict[str, ArrayType | FloatType]:
        all_vars: dict[str, ArrayType | FloatType] = {}

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
        u: ArrayType | None = None,
        p: ArrayType | None = None,
        q: ArrayType | None = None,
    ) -> ArrayType:
        arr_ = []
        u = self.u.est.last if u is None else u
        p = self.p.est.last if p is None else p
        q = self.q.est.last if q is None else q

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
        self.t.set_val(k=self.k, val=np.array([[self.k * self.dt]]))

    def create_data_preds_full(
        self,
        k_preds_full: PredDType | None = None,
        t_preds_full: PredDType | None = None,
        x_preds_full: PredDType | None = None,
        u_preds_full: PredDType | None = None,
        cost_x_full: PredDType | None = None,
        cost_y_full: PredDType | None = None,
        cost_u_full: PredDType | None = None,
        cost_du_full: PredDType | None = None,
    ) -> None:
        if k_preds_full is not None:
            self._data.predictions_full.k = k_preds_full

        if t_preds_full is not None:
            self._data.predictions_full.t = t_preds_full

        if x_preds_full is not None:
            self._data.predictions_full.x = x_preds_full

        if u_preds_full is not None:
            self._data.predictions_full.u = u_preds_full

        if cost_x_full is not None:
            self._data.predictions_full.cost_x = cost_x_full

        if cost_y_full is not None:
            self._data.predictions_full.cost_y = cost_y_full

        if cost_u_full is not None:
            self._data.predictions_full.cost_u = cost_u_full

        if cost_du_full is not None:
            self._data.predictions_full.cost_du = cost_du_full

    def get_var(self, var_: str) -> StateSpaceStructure:
        return self._physical_var(var_)

    def n(self, ss_var: str) -> int:
        return len(getattr(self._settings, f'{ss_var}_vars'))

    def make_step(
        self,
        u: Array2D | None = None,
        p: Array2D | None = None,
        q: Array2D | None = None,
    ) -> None:
        self._update_k_and_t()

        self.u.est.val = u if u is not None else self.u.est.last
        self.p.est.val = p if p is not None else self.p.est.last
        self.q.est.val = q if q is not None else self.q.est.last

        x_next, z_next, _, _ = self.compute_xz()

        self.x.est.val = x_next
        self.z.est.val = z_next

    def compute_xz(
        self,
        x0: Array2D | None = None,
        z0: Array2D | None = None,
        u: Array2D | None = None,
        p: Array2D | None = None,
        q: Array2D | None = None,
        t_grid: TgridType | None = None,
    ) -> tuple[Array2D, Array2D, Array2D, Array2D]:
        x0 = self.x.est.last if x0 is None else x0
        z0 = self.z.est.last if z0 is None else z0

        u = self.u.est.val if u is None else u
        p = self.p.est.val if p is None else p
        q = self.q.est.val if q is None else q

        upq_arr = self.get_upq(u=u, p=p, q=q)
        t_grid = [0, self._settings.dt] if t_grid is None else t_grid

        # logger.info(dict(x0=x0, z0=z0, upq_arr=upq_arr, t_grid=t_grid))
        x, z, x_arr, z_arr = self._integrator.integrate(
            x0=x0,
            z0=z0,
            upq_arr=upq_arr,
            t_grid=t_grid,
        )
        return x, z, x_arr, z_arr

    def export_data_csv(self, k0: int | None = None, kf: int | None = None):
        k0 = k0 if k0 is not None else 0
        kf = kf if kf is not None else self.k

        report_dir = Path().cwd() / 'report'
        report_dir.mkdir(exist_ok=True)

        dfs: list[pd.DataFrame] = [
            pd.DataFrame(np.arange(self.k + 1), columns=['k']),
            pd.DataFrame(self.t.hist[0, :], columns=['t']),
            pd.DataFrame(self.x.est.get_hist(k0, kf).T, columns=self.x_vars),
            pd.DataFrame(self.z.est.get_hist(k0, kf).T, columns=self.z_vars),
            pd.DataFrame(
                self.upq.est.get_hist(k0, kf).T, columns=self.upq_vars
            ),
        ]
        df_cost = self.predictions.df_cost

        df_merged = pd.concat(dfs, axis=1)
        df_merged.to_csv(report_dir / 'data.csv', index=False)


if __name__ == '__main__':
    pass
