from copy import copy
from typing import override

import casadi as ca
import numpy as np
from numpy.typing import NDArray

from nextpredco.core import Symbolic
from nextpredco.core.errors import StepSizeInitializationError
from nextpredco.core.integrator._integrator import IntegratorABC
from nextpredco.core.settings import TaylorSettings


class Taylor(IntegratorABC):
    def __init__(
        self,
        settings: TaylorSettings,
        equations: dict[str, Symbolic],
        h: float | None = None,
    ):
        super().__init__(settings, equations)
        if h is not None:
            self._settings.h = h

        if self._settings.h is None:
            raise StepSizeInitializationError()

        self._integrator = self._create_integrator()

    @override
    def _create_integrator(
        self,
        t_grid: list[float | int] | NDArray | None = None,
    ) -> ca.Function:
        x0: Symbolic = Symbolic.sym('x0', self._equations['x'].shape[0])
        z0: Symbolic = Symbolic.sym('z0', self._equations['z'].shape[0])
        p0: Symbolic = Symbolic.sym('p0', self._equations['p'].shape[0])

        if self._settings.order == 1:
            x = x0 + self._settings.h * self._ffunc(x0, z0, p0)

        return ca.Function('private_int', [x0, z0, p0], [x])

    @override
    def integrate(
        self,
        x0: Symbolic | NDArray,
        z0: Symbolic | NDArray,
        upq_arr: Symbolic | NDArray,
        t_grid: list[float | int] | NDArray | None = None,
    ):
        if self._settings.h is None:
            raise StepSizeInitializationError()

        # if t_grid is not None:
        #     min_ = round(t_grid[0] / self._settings.h)
        #     max_ = round(t_grid[-1] / self._settings.h)
        #     step = round(1 / self._settings.h)
        #     t_grid2 = np.arange(min_, max_ + step, step) * self._settings.h
        #     input(
        #         f't_grid = {t_grid}, t_grid2 = {t_grid2}, '
        #         'min_ = {min_}, max_ = {max_}, step = {step}'
        #     )

        # TODO, here we assume that the t_grid is divisible by h
        x_arr = []
        x = x0
        for k in range(upq_arr.shape[1]):
            x = self._integrator(x, z0, upq_arr[:, k])
            x_arr.append(copy(x))

        if isinstance(x0, Symbolic):
            return (x, np.array([[]]).T, ca.hcat(x_arr), np.array([[]]).T)

        return (x, np.array([[]]).T, np.hstack(x_arr), np.array([[]]).T)

    @staticmethod
    def _is_divisible(x: float, y: float, tol: float = 1e-9) -> bool:
        return abs(x % y) < tol
