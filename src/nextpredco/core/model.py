import contextlib
import importlib.util
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, override

import casadi as ca
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nextpredco.core import utils
from nextpredco.core.consts import (
    CONFIG_FOLDER,
    DATA_DIR,
    SS_VARS_DB,
    SS_VARS_PRIMARY,
    SS_VARS_SECONDARY,
    SS_VARS_SOURCES,
)
from nextpredco.core.custom_types import SourceType, SymVar
from nextpredco.core.descriptors import (
    ReadOnlyFloat,
    ReadOnlyInt,
    ReadOnlyPandas,
    ReadOnlySource,
    SystemVariable,
    TimeVariable,
)
from nextpredco.core.errors import SystemVariableError
from nextpredco.core.integrator import IDASIntegrator
from nextpredco.core.logger import logger
from nextpredco.core.settings.settings import IDASSettings, ModelSettings

try:
    from rich.pretty import pretty_repr
except ImportError:

    def pretty_repr(obj: Any) -> Any:  # type: ignore[misc]
        return obj


from collections.abc import Callable


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
    k = ReadOnlyInt()
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
    upq = SystemVariable()
    t = TimeVariable()
    t_clock = TimeVariable()

    def __init__(
        self,
        # model_data_path: Path = DATA_DIR / "ex_chen1998.csv",
        settings: ModelSettings | None = None,
        integrator_settings: IDASSettings | None = None,
    ) -> None:  # Extract model information and initial values
        self._settings = settings
        logger.debug(
            'Model settings:\n%s',
            pretty_repr(self._settings),
        )

        self._integrator: IDASIntegrator | None = None
        if integrator_settings is not None:
            self._integrator = IDASIntegrator(integrator_settings)
            logger.debug(
                'Integrator settings:\n%s',
                pretty_repr(integrator_settings),
            )

        self._data = self._create_data()
        logger.debug(
            'Created data holder. Result:\n%s',
            pretty_repr(self._data),
        )

        self._transient_eqs, self._transient_funcs = self._create_equations()
        logger.debug('Loaded equations.')

    @staticmethod
    def _load_equations() -> tuple[
        Callable[..., SymVar],
        Callable[..., SymVar],
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

    def _create_upq_dict(
        self,
        upq: NDArray | SymVar,
    ) -> dict[str, NDArray | SymVar]:
        # TODO: Review the type hints
        upq_dict = {}

        for var_ in self._settings.upq_vars:
            idx = self._settings.upq_vars.index(var_)
            upq_dict[var_] = upq[idx, 0]

        return upq_dict

    def _create_equations(
        self,
    ) -> tuple[dict[str, SymVar], dict[str, ca.Function]]:
        # Create symbolic variables
        x = SymVar.sym('x', self.n('x'))
        z = SymVar.sym('z', self.n('z'))
        upq = SymVar.sym('upq', self.n('upq'))

        # Create dictionary with all symbolic variables
        all_vars: dict[str, SymVar] = {}
        for var_ in self._settings.x_vars:
            all_vars[var_] = x[self._settings.x_vars.index(var_), 0]

        for var_ in self._settings.z_vars:
            all_vars[var_] = z[self._settings.z_vars.index(var_), 0]

        for var_ in self._settings.upq_vars:
            all_vars[var_] = upq[self._settings.upq_vars.index(var_), 0]

        # Load equations from equations.py
        create_f, create_g = self._load_equations()
        f = create_f(**all_vars)
        f_func = ca.Function('f', [x, z, upq], [f])

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
            g_func = ca.Function('g', [x, z, upq], [g])
            transient_eqs['alg'] = g
            transient_funcs['f'] = g_func

        return transient_eqs, transient_funcs
        # print(f"f  = {type(f)}")

    # @abstractmethod
    # @staticmethod
    # def create_f(**kwargs) -> SymVar:
    #     pass

    # # @abstractmethod
    # @staticmethod
    # def create_g(**kwargs) -> SymVar:
    #     pass

    def create_integrator(
        self,
        t0: float,
        t_grid: NDArray[np.float64] | list[float],
        opts: dict[str, float | str],
    ) -> ca.Function:
        # Choose Idas by default
        return ca.integrator(
            'integrator',
            'idas',
            self._transient_eqs,
            t0,
            t_grid,
            opts,
        )

    def _update_k(self) -> None:
        self._data.k += 1

        # TODO: try other ways to update time
        # Try case to work with t_clock
        self.t.val[0, 0] = self._data.k * self._settings.dt

    def n(self, ss_var: str) -> int:
        return len(getattr(self._settings, f'{ss_var}_vars'))

    def make_step(self) -> None:
        # x, z, _, _ = self.compute_xz(
        #     t0=self.k,
        #     t_grid=[self.k, self.k + 1],
        #     opts={'abstol': 1e-6, 'reltol': 1e-6},
        #     x0=self.x.est.val,
        #     z0=self.z.est.val,
        #     upq=self.upq.est.val,
        # )
        x, z, _, _ = self._integrator.integrate(
            equations=self._transient_eqs,
            x0=self.x.est.val,
            z0=self.z.est.val,
            t0=self.k,
            t_grid=[self.k, self.k + 1],
            p0=self.upq.est.val,
        )
        self._update_k()
        self.x.est.val = x
        self.z.est.val = z

    def compute_xz(
        self,
        t0: float,
        t_grid: NDArray[np.float64] | list[float],
        opts: dict[str, float | str],
        x0: NDArray,
        z0: NDArray,
        upq: NDArray,
    ):
        integrator = self.create_integrator(t0=t0, t_grid=t_grid, opts=opts)
        sol = integrator(x0=x0, z0=z0, p=upq)
        x_arr = sol['xf'].full()
        z_arr = sol['zf'].full()

        # Get final state vector
        x = x_arr[:, -1, None]
        z = z_arr[:, -1, None]

        return x, z, x_arr, z_arr


if __name__ == '__main__':
    # model = Model()

    # print(f"\nInitial condition\n\tmodel.x.est.val = {model.x.est.val.T}")
    # model.compute_xz(
    #     t0=0,
    #     t_grid=[0, 1, 2],
    #     opts={"abstol": 1e-6, "reltol": 1e-6},
    #     x0=model.x.est.val,
    #     z0=model.z.est.val,
    #     upq=model.upq.est.val,
    # )
    pass
