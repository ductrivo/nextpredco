from abc import ABC, abstractmethod

from numpy.typing import NDArray

from nextpredco.core.integrator import IntegratorFactory, IntegratorSettings
from nextpredco.core.model import Model
from nextpredco.core.optimizer import Optimizer, OptimizerFactory
from nextpredco.core.settings import (
    ControllerSettings,
    OptimizerSettings,
)


class ControllerABC(ABC):
    def __init__(
        self,
        settings: ControllerSettings,
        model: Model | None = None,
        optimizer_settings: OptimizerSettings | None = None,
        integrator_settings: IntegratorSettings | None = None,
    ):
        self._model = model
        self._settings = settings
        self._settings.dt = model.dt if settings.dt == -1 else settings.dt

        self._optimizer = (
            OptimizerFactory.create(optimizer_settings)
            if optimizer_settings is not None
            else None
        )

        if integrator_settings is not None:
            if integrator_settings.h == -1:
                integrator_settings.h = self._settings.dt

            self._integrator = IntegratorFactory.create(
                settings=integrator_settings,
                equations=model._integrator._equations,
            )
        else:
            self._integrator = None

        self._n_ctrl = round(self._settings.dt / self._model.dt)

    @abstractmethod
    def make_step(self) -> NDArray:
        pass

    @property
    def model(self) -> Model | None:
        return self._model

    @property
    def optimizer(self) -> Optimizer | None:
        return self._optimizer
