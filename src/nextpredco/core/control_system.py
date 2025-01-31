import importlib.util
from pathlib import Path
from typing import override

import pandas as pd
from numpy.typing import NDArray

from nextpredco.core import utils
from nextpredco.core.consts import CONFIG_FOLDER, SETTING_FOLDER
from nextpredco.core.controller import MPC, PID, Controller
from nextpredco.core.custom_types import SourceType
from nextpredco.core.integrator import IDAS
from nextpredco.core.model import Model
from nextpredco.core.observer import Observer
from nextpredco.core.optimizer import IPOPT
from nextpredco.core.plant import Plant
from nextpredco.core.settings.settings import (
    ControllerSettings,
    IDASSettings,
    IPOPTSettings,
    KalmanSettings,
    ModelSettings,
    MPCSettings,
    ObserverSettings,
    PIDSettings,
    PlantSettings,
    SettingsFactory,
    extract_settings_from_file,
)


class ControlSystem:
    def __init__(self):
        self.model: Model | None = None  # type: ignore[annotation-unchecked]
        self.controller = None
        self.observer = None
        self.plant = None

    def simulate(
        self, u_arr: NDArray, p_arr: NDArray, q_arr: NDArray | None = None
    ):
        for k in range(u_arr.shape[1]):
            self.model.make_step(
                u=u_arr[:, k, None],
                p=p_arr[:, k, None],
                # q=q_arr[:, k],
            )


class ControlSystemBuilder:
    def __init__(self):
        self.system = ControlSystem()

    def set_model(
        self,
        settings: ModelSettings,
        integrator_settings: IDASSettings | None,
    ) -> 'ControlSystemBuilder':
        # Create integrator if settings are provided
        if integrator_settings is not None:
            integrator = IDAS(integrator_settings)
        else:
            integrator = None

        # Create model
        self.system.model = Model(settings, integrator)
        return self

    def set_controller(
        self,
        settings: PIDSettings | MPCSettings,
        optimizer_settings: IPOPTSettings | None = None,
    ) -> 'ControlSystemBuilder':
        # Create optimizer if settings are provided
        if optimizer_settings is not None:
            # input(optimizer_settings)
            optimizer = IPOPT(optimizer_settings)

        # Create controller based on the setting type
        if isinstance(settings, PIDSettings):
            self.system.controller = PID(settings)

        elif isinstance(settings, MPCSettings):
            self.system.controller = MPC(
                settings=settings,
                model=self.system.model,
                optimizer=optimizer,
            )
        return self

    def set_observer(self, settings: KalmanSettings) -> 'ControlSystemBuilder':
        # if isinstance(settings, KalmanSettings):
        #     self.system.observer = Observer(settings)
        return self

    def set_plant(self, plant: Plant) -> 'ControlSystemBuilder':
        self.system.plant = plant
        return self

    def build(self) -> ControlSystem:
        return self.system


class Director:
    def __init__(self, builder: ControlSystemBuilder):
        self.builder = builder

    def construct(self) -> ControlSystem:
        settings = extract_settings_from_file()
        utils.print(settings)
        self.builder.set_model(settings['model'], settings['model_integrator'])

        if 'controller' in settings:
            self.builder.set_controller(
                settings['controller'],
                settings['controller_optimizer'],
            )

        if 'observer' in settings:
            self.builder.set_observer(settings['observer'])

        if 'plant' in settings:
            self.builder.set_plant(settings['plant'])
        return self.builder.build()


def construct_control_system() -> ControlSystem:
    builder = ControlSystemBuilder()
    director = Director(builder)
    return director.construct()
