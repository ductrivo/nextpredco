from abc import ABC, abstractmethod
from typing import override

from numpy.typing import NDArray

from nextpredco.core.descriptors import ReadOnlyFloat, ReadOnlyInt
from nextpredco.core.logger import logger
from nextpredco.core.model import Model
from nextpredco.core.optimizer import IPOPT, Optimizer
from nextpredco.core.settings.settings import (
    ControllerSettings,
    IDASSettings,
    MPCSettings,
    PIDSettings,
)


class Controller(ABC):
    def __init__(
        self,
        settings: ControllerSettings,
        model: Model | None = None,
        optimizer: IPOPT | None = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._dt = settings.dt

        self._n_ctrl = round(self._dt / self._model.dt)

    @abstractmethod
    def make_step(self) -> NDArray:
        pass

    @property
    def model(self) -> Model | None:
        return self._model

    @property
    def optimizer(self) -> IPOPT | None:
        return self._optimizer


class PID(Controller):
    def __init__(
        self,
        pid_settings: PIDSettings,
    ):
        super().__init__(pid_settings)
        self._kp = pid_settings.kp
        self._ki = pid_settings.ki
        self._kd = pid_settings.kd

    @override
    def make_step(self):
        pass

    def auto_tuning(self):
        pass


class MPC(Controller):
    n_pred = ReadOnlyInt()

    def __init__(
        self,
        settings: MPCSettings,
        model: Model,
        optimizer: IPOPT | None,
    ):
        super().__init__(settings, model, optimizer)

        self._n_pred = settings.n_pred
        # logger.debug(optimizer.settings)
        # input('Press Enter to continue...')

    @override
    def make_step(self):
        pass
