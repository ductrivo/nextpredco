import importlib.util
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import reduce as reduce
from pathlib import Path
from typing import override

import casadi as ca
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from rich.pretty import pretty_repr

from nextpredco.core._consts import CONFIG_FOLDER, COST_ELEMENTS, SS_VARS_DB
from nextpredco.core._data import ModelData
from nextpredco.core._element import ElementABC
from nextpredco.core._logger import logger
from nextpredco.core._sync import GlobalState
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
from nextpredco.core.integrator import (
    Integrator,
    IntegratorFactory,
    IntegratorSettings,
)
from nextpredco.core.model._descriptors import (
    ReadOnly,
    # SystemCallable,
    # SystemPredictions,
    SystemTime,
    SystemVariable,
    VariableView,
)
from nextpredco.core.settings import ModelSettings


class ModelABC(ElementABC):
    t = SystemTime()
    # predictions = SystemPredictions()
    # _physical_var = SystemCallable()
    integrator = ReadOnly[Integrator]()

    settings = ReadOnly[ModelSettings]()
    data = ReadOnly[ModelData]()

    dt = ReadOnly[FloatType]('settings')
    t_max = ReadOnly[FloatType]('settings')

    # k = ReadOnly[IntType]('data')
    k_max = ReadOnly[IntType]('data')

    x = SystemVariable()
    z = SystemVariable()
    u = SystemVariable()
    p = SystemVariable()
    q = SystemVariable()
    m = SystemVariable()
    y = SystemVariable()
    upq = SystemVariable()

    def __init__(
        self,
        settings: ModelSettings,
        integrator_settings: IntegratorSettings | None = None,
    ):
        super().__init__()

        # Load settings
        self._settings = settings
        logger.debug('Model settings:\n%s', pretty_repr(self._settings))

        # Load data
        GlobalState.create(self._settings)
        self._data = self._create_data()

        # Load equations
        self._equations, self.transient_funcs = self._create_equations()

        # Load integrator
        if integrator_settings is not None:
            self._integrator = IntegratorFactory.create(
                settings=integrator_settings,
                equations=self._equations,
                h=GlobalState.dt(),
            )
        else:
            self._integrator = None

        self._inputs = {'u': [self.u, 'next']}
        self._outputs = {
            'm': [self.m, 'vec'],
            'u': [self.u, 'vec'],
            'p': [self.p, 'vec'],
            'q': [self.q, 'vec'],
        }

    def _create_data(self) -> ModelData:
        # Create data holder
        data: dict[str, IntType | Array2D | OrderedDict] = {}
        # input(self._settings)
        # Extract time information
        data['k'] = 0
        k_max = round(GlobalState.t_max() / GlobalState.dt())
        data['k_max'] = k_max

        # Create data holder (full arrays) for each source
        created_attrs: list[str] = []
        for ss_var in SS_VARS_DB:
            vars_list = getattr(self._settings, f'{ss_var}_vars')
            GlobalState.set_vars(name=ss_var, val=vars_list)

            length = len(vars_list)
            if length > 0:
                init_val = np.array(
                    [[self._settings.info[key] for key in vars_list]],
                ).T
            else:
                init_val = np.zeros((length, 1))

            arr_full = np.zeros((init_val.shape[0], k_max + 1))
            arr_full[:, 0, None] += init_val
            data[f'{ss_var}_full'] = arr_full

            # TODO: Modify the settings structure to get the max and min values
            # Check if data changes.
            min_vals: OrderedDict[IntType, Array2D] = OrderedDict()
            max_vals: OrderedDict[IntType, Array2D] = OrderedDict()

            min_vals[0] = init_val
            max_vals[0] = init_val * 1.5

            data[f'{ss_var}_min'] = min_vals
            data[f'{ss_var}_max'] = max_vals

        # Create data holder (full arrays) for time variables
        data['t_full'] = np.zeros((1, k_max + 1))
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

        for var in GlobalState.upq_vars():
            idx = GlobalState.upq_vars().index(var)
            upq_dict[var] = upq[idx, 0]

        return upq_dict

    def _create_equations(
        self,
    ) -> tuple[dict[str, Symbolic], dict[str, ca.Function]]:
        # Create symbolic variables
        x = Symbolic.sym('__x', GlobalState.n('x'))
        z = Symbolic.sym('__z', GlobalState.n('z'))
        upq = Symbolic.sym('__upq', GlobalState.n('upq'))

        # Create dictionary with all symbolic variables
        all_vars = self.get_all_vars(x=x, z=z, upq=upq)
        logger.debug(pretty_repr(all_vars))

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

        if GlobalState.n('z') > 0:
            g = create_g(**all_vars)
            g_func = ca.Function('private_func_g', [x, z, upq], [g])
            transient_eqs['alg'] = g
            # transient_funcs['f'] = g_func

        return transient_eqs, transient_funcs

    def get_all_vars(
        self,
        **ss_vars: ArrayType,
    ):
        # all_vars: dict[str, ArrayType | FloatType] = {}

        all_vars = self._settings.info.copy()

        for ss_var in ['x', 'z', 'upq']:
            vars_list: list[str] = getattr(GlobalState, f'{ss_var}_vars')()
            for var in vars_list:
                arr = ss_vars[ss_var]
                idx = vars_list.index(var)
                all_vars[var] = arr[idx, 0]

        logger.debug(f'Inside: {pretty_repr(all_vars)}')
        return all_vars

    def get_upq(
        self,
        u_arr: ArrayType | None = None,
        p_arr: ArrayType | None = None,
        q_arr: ArrayType | None = None,
    ):
        u_arr = self.u.last if u_arr is None else u_arr
        p_arr = self.p.last if p_arr is None else p_arr
        q_arr = self.q.last if q_arr is None else q_arr

        n_cols = max(arr.shape[1] for arr in [u_arr, p_arr, q_arr])

        u_arr = (
            np.repeat(u_arr, n_cols, axis=1)
            if u_arr.shape[1] != n_cols
            else u_arr
        )
        p_arr = (
            np.repeat(p_arr, n_cols, axis=1)
            if p_arr.shape[1] != n_cols
            else p_arr
        )
        q_arr = (
            np.repeat(q_arr, n_cols, axis=1)
            if q_arr.shape[1] != n_cols
            else q_arr
        )

        arr: list[ArrayType] = []
        for var in GlobalState.upq_vars():
            if var in GlobalState.u_vars():
                idx = GlobalState.u_vars().index(var)
                arr.append(u_arr[[idx], :])
            elif var in GlobalState.p_vars():
                idx = GlobalState.p_vars().index(var)
                arr.append(p_arr[[idx], :])
            elif var in GlobalState.q_vars():
                idx = GlobalState.q_vars().index(var)
                arr.append(q_arr[[idx], :])

        if (
            isinstance(u_arr, Symbolic)
            or isinstance(p_arr, Symbolic)
            or isinstance(q_arr, Symbolic)
        ):
            return ca.vcat(arr)

        return np.vstack(arr)

    @override
    def make_step(
        self,
        u: Array2D | None = None,
        p: Array2D | None = None,
        q: Array2D | None = None,
    ) -> None:
        # self._update_k_and_t()
        GlobalState.update_k()
        self.t.vec = np.array([[GlobalState.k() * self.dt]])

        self.u.vec = u if u is not None else self.u.last
        self.p.vec = p if p is not None else self.p.last
        self.q.vec = q if q is not None else self.q.last

        x_next, z_next, _, _ = self.compute_xz()

        self.x.vec = x_next
        self.z.vec = z_next

    # def _update_k_and_t(self) -> None:
    #     # self._data.k += 1
    #     # TODO: try other ways to update time
    #     self.t.set_val(
    #         k=GlobalState.k(),
    #         val=np.array([[GlobalState.k() * self.dt]]),
    #     )

    def compute_xz(
        self,
        x0: Array2D | None = None,
        z0: Array2D | None = None,
        u_arr: Array2D | None = None,
        p_arr: Array2D | None = None,
        q_arr: Array2D | None = None,
        t_grid: TgridType | None = None,
    ) -> tuple[Array2D, Array2D, Array2D, Array2D]:
        x0 = self.x.last if x0 is None else x0
        z0 = self.z.last if z0 is None else z0

        u_arr = self.u.vec if u_arr is None else u_arr
        p_arr = self.p.vec if p_arr is None else p_arr
        q_arr = self.q.vec if q_arr is None else q_arr

        upq_arr = self.get_upq(u_arr=u_arr, p_arr=p_arr, q_arr=q_arr)
        t_grid = [0, GlobalState.dt()] if t_grid is None else t_grid

        # logger.info(
        #     pretty_repr(
        #         dict(
        #             name=self._integrator._settings.name,
        #             x0=x0.T,
        #             z0=z0,
        #             u_arr=u_arr.T,
        #             p_arr=p_arr.T,
        #             q_arr=q_arr,
        #             upq_arr=upq_arr.T,
        #             t_grid=t_grid,
        #             diff=self.transient_funcs['f'](x0, z0, upq_arr[:, [0]])
        #             * GlobalState.dt(),
        #         ),
        #     )
        # )
        x, z, x_arr, z_arr = self._integrator.integrate(
            x0=x0,
            z0=z0,
            upq_arr=upq_arr,
            t_grid=t_grid,
        )

        # logger.info(
        #     pretty_repr(
        #         dict(
        #             x_arr=x_arr.T,
        #         ),
        #     )
        # )

        # input('compute_xz, integrate')
        return x, z, x_arr, z_arr

    def get_var(self, var: str) -> VariableView:
        if var in GlobalState.x_vars():
            return VariableView(
                k=GlobalState.k(),
                full_data=self._data.x_full,
                indexes=[GlobalState.x_vars().index(var)],
                columns=[var],
            )

        if var in GlobalState.z_vars():
            return VariableView(
                k=GlobalState.k(),
                full_data=self._data.z_full,
                indexes=[GlobalState.z_vars().index(var)],
                columns=[var],
            )

        if var in GlobalState.upq_vars():
            return VariableView(
                k=GlobalState.k(),
                full_data=self._data.upq_full,
                indexes=[GlobalState.upq_vars().index(var)],
                columns=[var],
            )

        msg = f"Variable '{var}' not found in x_vars, z_vars, or upq_vars."
        raise ValueError(msg)
