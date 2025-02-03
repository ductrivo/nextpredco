from abc import ABC, abstractmethod
from copy import copy
from typing import override

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from nextpredco.core.custom_types import Symbolic
from nextpredco.core.descriptors import ReadOnlyInt
from nextpredco.core.logger import logger
from nextpredco.core.model import Model
from nextpredco.core.optimizer import IPOPT
from nextpredco.core.settings import (
    ControllerSettings,
    MPCSettings,
    PIDSettings,
)


class Controller(ABC):
    def __init__(
        self,
        settings: ControllerSettings | MPCSettings | PIDSettings,
        model: Model | None = None,
        optimizer: IPOPT | None = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._settings = settings
        self._settings.dt = model.dt if settings.dt == -1 else settings.dt

        self._n_ctrl = round(self._settings.dt / self._model.dt)

    @abstractmethod
    def make_step(self) -> NDArray:
        pass

    @property
    def model(self) -> Model | None:
        return self._model

    @property
    def optimizer(self) -> IPOPT | None:
        return self._optimizer
