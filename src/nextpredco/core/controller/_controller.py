from abc import ABC, abstractmethod

from numpy.typing import NDArray

from nextpredco.core._element import ElementABC
from nextpredco.core._sync import GlobalState
from nextpredco.core.integrator import IntegratorFactory, IntegratorSettings
from nextpredco.core.model._descriptors import ReadOnly, VariableView
from nextpredco.core.model._model import ModelABC
from nextpredco.core.optimizer import Optimizer, OptimizerFactory
from nextpredco.core.settings import (
    ControllerSettings,
    OptimizerSettings,
)


class ControllerABC(ElementABC, ABC):
    model = ReadOnly[ModelABC]()

    @property
    def x(self) -> VariableView:
        return self._model.x

    @property
    def z(self) -> VariableView:
        return self._model.z

    @property
    def u(self) -> VariableView:
        return self._model.u

    @property
    def p(self) -> VariableView:
        return self._model.p

    @property
    def q(self) -> VariableView:
        return self._model.q

    def __init__(
        self,
        settings: ControllerSettings,
        model: ModelABC | None = None,
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
                equations=model.integrator.equations,
            )
        else:
            self._integrator = None

        # self._n_ctrl = round(self._settings.dt / self._model.dt)

        # self.inputs = {'u': self.u}
        # self.outputs = {'u': self.u}
