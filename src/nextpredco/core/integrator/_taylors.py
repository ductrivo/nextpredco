from copy import copy
from typing import override

import casadi as ca
import numpy as np

from nextpredco.core._typing import ArrayType, Symbolic, TgridType
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
        t_grid: TgridType | None = None,
    ) -> ca.Function:
        x0: Symbolic = Symbolic.sym('x0', self._equations['x'].shape[0])
        z0: Symbolic = Symbolic.sym('z0', self._equations['z'].shape[0])
        p0: Symbolic = Symbolic.sym('p0', self._equations['p'].shape[0])

        h = self._settings.h if t_grid is None else t_grid[1] - t_grid[0]
        if self._settings.order == 1:
            x = x0 + h * self._ffunc(x0, z0, p0)

        return ca.Function('private_int', [x0, z0, p0], [x])

    @override
    def integrate(
        self,
        x0: ArrayType,
        z0: ArrayType,
        upq_arr: ArrayType,
        t_grid: TgridType | None = None,
    ) -> tuple[ArrayType, ArrayType, ArrayType, ArrayType]:
        if self._settings.h is None:
            raise StepSizeInitializationError()

        t_grid = t_grid if t_grid is not None else [0, self._settings.h]

        if len(list(t_grid)) != upq_arr.shape[1] + 1:
            msg = 'The length of t_grid must be equal to upq_arr.shape[1] + 1.'
            raise ValueError(msg)

        if t_grid is not None:
            min_ = np.floor(t_grid[0] / self._settings.h)
            max_ = np.floor(t_grid[-1] / self._settings.h)
            t_grid2 = np.arange(min_, max_ + 1) * self._settings.h

        x_arr = []
        x: ArrayType | ca.DM = x0
        k = 0
        for i in range(len(t_grid2) - 1):
            dt = t_grid2[i + 1] - t_grid2[i]

            if self._is_greater(self._settings.h, dt):
                h_new = self._settings.h - dt
                integrator = self._create_integrator([0, h_new])
                x = integrator(x, z0, upq_arr[:, k, None])
            else:
                x = self._integrator(x, z0, upq_arr[:, k])

            if self._is_in_grid(t=t_grid2[i + 1], t_grid=t_grid):
                x_arr.append(copy(x))
                k += 1

        if isinstance(x0, Symbolic):
            return (x, np.array([[]]).T, ca.hcat(x_arr), np.array([[]]).T)

        if isinstance(x, ca.DM):
            x = x.full()
        return (x, np.array([[]]).T, np.hstack(x_arr), np.array([[]]).T)

    @staticmethod
    def _is_divisible(x: float, y: float, tol: float = 1e-6) -> bool:
        return abs(x % y) < tol

    @staticmethod
    def _is_greater(x: float, y: float, tol: float = 1e-6) -> bool:
        return x - y > tol

    @staticmethod
    def _is_in_grid(t: float, t_grid: TgridType, tol: float = 1e-6) -> bool:
        return any(np.isclose(t, tg, atol=tol) for tg in t_grid)
