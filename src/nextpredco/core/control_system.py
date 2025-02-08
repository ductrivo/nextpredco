import numpy as np
from numpy.typing import NDArray

from nextpredco.core import tools
from nextpredco.core._logger import logger
from nextpredco.core._typing import TgridType
from nextpredco.core.controller import Controller, ControllerFactory
from nextpredco.core.model import Model, Plant
from nextpredco.core.settings import (
    ControllerSettings,
    IntegratorSettings,
    ModelSettings,
    ObserverSettings,
    OptimizerSettings,
    get_settings,
)


class ControlSystem:
    def __init__(self):
        self.model: Model | None = None  # type: ignore[annotation-unchecked]
        self.controller: Controller | None = None  # type: ignore[annotation-unchecked]
        self.observer = None
        self.plant = None

    def simulate_model_only(
        self,
        t_grid: TgridType,
        x_arr: NDArray,
        u_arr: NDArray | None = None,
        p_arr: NDArray | None = None,
        q_arr: NDArray | None = None,
    ):
        if u_arr is not None:
            k_max = u_arr.shape[1]
        elif p_arr is not None:
            k_max = p_arr.shape[1]
        elif q_arr is not None:
            k_max = q_arr.shape[1]
        else:
            msg = 'At least one of u_arr, p_arr, or q_arr must be provided'
            raise ValueError(msg)

        x = x_arr[:, 0, None]
        x_results: list[NDArray] = [x]
        for k in range(k_max):
            # print(f'k = {k}')
            u = None if u_arr is None else u_arr[:, k, None]
            p = None if p_arr is None else p_arr[:, k, None]
            q = None if q_arr is None else q_arr[:, k, None]
            while True:
                self.model.make_step(u=u, p=p, q=q)
                if tools.is_in_list(t=self.model.t.val[0, 0], t_grid=t_grid):
                    break

            x_results.append(x)

        x_results_ = np.hstack(x_results)
        return x_results_, None

    def simulate(self):
        logger.info(
            'Total size of ModelData: %s MB.', self.model._data.size / 1024
        )
        # input('Press Enter to start simulation')

        for k in range(self.model.k_max):
            # print(f'k = {k}')
            self.model.y.goal.val = 0.6
            u = self.controller.make_step()
            self.model.make_step(u=u)


class ControlSystemBuilder:
    def __init__(self):
        self.system = ControlSystem()

    def set_model(
        self,
        settings: ModelSettings,
        integrator_settings: IntegratorSettings | None = None,
    ) -> 'ControlSystemBuilder':
        # Create model
        self.system.model = Model(settings, integrator_settings)
        return self

    def set_controller(
        self,
        settings: ControllerSettings,
        optimizer_settings: OptimizerSettings | None = None,
        integrator_settings: IntegratorSettings | None = None,
    ) -> 'ControlSystemBuilder':
        # Create controller based on the setting type
        self.system.controller = ControllerFactory.create(
            settings=settings,
            model=self.system.model,
            optimizer_settings=optimizer_settings,
            integrator_settings=integrator_settings,
        )

        return self

    def set_observer(
        self,
        settings: ObserverSettings,
        optimizer_settings: OptimizerSettings | None = None,
        integrator_settings: IntegratorSettings | None = None,
    ) -> 'ControlSystemBuilder':
        return self

    def set_plant(self, plant: Plant) -> 'ControlSystemBuilder':
        self.system.plant = plant
        return self

    def build(self) -> ControlSystem:
        return self.system


class Director:
    def __init__(self, builder: ControlSystemBuilder):
        self.builder = builder

    def construct(
        self,
        settings: dict,
    ) -> ControlSystem:
        self.builder.set_model(settings['model'], settings['model.integrator'])

        if 'controller' in settings:
            self.builder.set_controller(
                settings=settings['controller'],
                optimizer_settings=settings['controller.optimizer'],
                integrator_settings=settings['controller.integrator'],
            )

        if 'observer' in settings:
            self.builder.set_observer(
                settings=settings['observer'],
                optimizer_settings=settings['observer.optimizer'],
                integrator_settings=settings['observer.integrator'],
            )

        if 'plant' in settings:
            self.builder.set_plant(settings['plant'])
        return self.builder.build()


def construct_control_system(
    setting_file_name: str = 'settings_template.csv',
) -> ControlSystem:
    # To construct a control system,
    # we need to read the settings from a CSV file
    settings = get_settings(setting_file_name)

    builder = ControlSystemBuilder()
    director = Director(builder)
    return director.construct(settings)
