from abc import ABC, abstractmethod

import casadi as ca
from numpy.typing import NDArray

from nextpredco.core import Symbolic
from nextpredco.core.settings import (
    IntegratorSettings,
    TaylorSettings,
)


class IntegratorABC(ABC):
    def __init__(
        self,
        settings: IntegratorSettings | TaylorSettings,
        equations: dict[str, Symbolic],
    ):
        super().__init__()
        self._settings = settings
        self._equations = equations
        self._ffunc, self._gfunc = self._create_funcs(equations)

    @property
    def opts(self) -> dict:
        if hasattr(self._settings, 'opts'):
            return self._settings.opts

        # TODO: Better return None?
        return {}

    @property
    def integrator(self) -> ca.Function | None:
        if hasattr(self, '_integrator'):
            return self._integrator
        return None

    def _create_funcs(
        self,
        equations: dict[str, Symbolic],
        # x0: Symbolic,
        # z0: Symbolic,
        # p0: Symbolic,
    ) -> tuple[ca.Function, ca.Function | None]:
        x = equations['x']
        z = equations['z']
        p = equations['p']
        f = equations['ode']
        f_func = ca.Function('private_f_func', [x, z, p], [f])

        if 'alg' in equations:
            g = equations['alg']
            g_func = ca.Function('private_g_func', [x, z, p], [g])
            return f_func, g_func
        return f_func, None

    @abstractmethod
    def integrate(
        self,
        x0: Symbolic | NDArray,
        z0: Symbolic | NDArray,
        upq_arr: Symbolic | NDArray,
        t_grid: list[float | int] | NDArray | None = None,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        pass

    @abstractmethod
    def _create_integrator(
        self,
        t_grid: list[float | int] | NDArray | None = None,
    ) -> ca.Function:
        pass
