from abc import ABC, abstractmethod

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from nextpredco.core.custom_types import SymVar
from nextpredco.core.settings.settings import IDASSettings


class Integrator(ABC):
    @abstractmethod
    def integrate(
        self,
        equations: dict[str, SymVar],
        x0: NDArray,
        t0: float,
        t_grid=ArrayLike,
        z0: NDArray | None = None,
        p0: NDArray | None = None,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        pass


class IDASIntegrator(Integrator):
    def __init__(self, settings: IDASSettings):
        self.opts = settings.opts

    def integrate(
        self,
        equations: dict[str, SymVar],
        x0: NDArray,
        t0: float,
        t_grid=ArrayLike,
        z0: NDArray | None = None,
        p0: NDArray | None = None,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        integrator = ca.integrator(
            'integrator',
            'idas',
            equations,
            t0,
            t_grid,
            self.opts,
        )

        sol = integrator(x0=x0, z0=z0, p=p0)
        x_arr = sol['xf'].full()
        z_arr = sol['zf'].full()

        # Get final state vector
        x = x_arr[:, -1, None]
        z = z_arr[:, -1, None]

        return x, z, x_arr, z_arr
