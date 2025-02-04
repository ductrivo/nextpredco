from abc import ABC, abstractmethod

import casadi as ca
from numpy.typing import ArrayLike, NDArray

from nextpredco.core import Symbolic, logger
from nextpredco.core.settings import IDASSettings


class Integrator(ABC):
    @abstractmethod
    def integrate(
        self,
        equations: dict[str, Symbolic],
        x0: NDArray,
        t_grid=ArrayLike,
        z0: NDArray | None = None,
        p0: NDArray | None = None,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        pass
