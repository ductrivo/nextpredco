from copy import copy
from typing import override

import casadi as ca
import numpy as np
from numpy.typing import NDArray

from nextpredco.core._typing import ArrayType, Symbolic, TgridType
from nextpredco.core.errors import StepSizeInitializationError
from nextpredco.core.integrator._integrator import IntegratorABC
from nextpredco.core.settings import IDASSettings


class IDAS(IntegratorABC):
    def __init__(
        self,
        settings: IDASSettings,
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
        if t_grid is None:
            return ca.integrator(
                'integrator0',
                'idas',
                self._equations,
                0,
                self._settings.h,
                self._settings.opts,
            )
        return ca.integrator(
            'integrator',
            'idas',
            self._equations,
            t_grid[0],
            t_grid,
            self._settings.opts,
        )

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

        if t_grid is None:
            t_grid = [0, self._settings.h]

        if len(t_grid) == 2 and np.isclose(
            t_grid[1] - t_grid[0], self._settings.h
        ):
            integrator = self._integrator
        else:
            integrator = self._create_integrator(t_grid)

        if (
            isinstance(x0, Symbolic)
            or isinstance(z0, Symbolic)
            or isinstance(upq_arr, Symbolic)
        ):
            raise NotImplementedError('Symbolic integration is not supported.')

        if len(list(t_grid)) != upq_arr.shape[1] + 1:
            msg = 'The length of t_grid must be equal to upq_arr.shape[1] + 1.'
            raise ValueError(msg)

        x, z = x0, z0
        x_list: list[NDArray] = []
        z_list: list[NDArray] = []

        for k in range(upq_arr.shape[1]):
            upq = upq_arr[:, k]
            sol = integrator(x0=x0, z0=z0, p=upq)
            x = sol['xf'].full()
            z = sol['zf'].full()
            x_list.append(copy(x))
            z_list.append(copy(z))

        x_arr = np.hstack(x_list)
        z_arr = np.hstack(z_list)

        # Get final state vector
        x = x_arr[:, -1, None]
        z = z_arr[:, -1, None]

        return x, z, x_arr, z_arr
