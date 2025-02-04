from numpy.typing import NDArray

from nextpredco.core import tools
from nextpredco.core.controller import MPC, PID
from nextpredco.core.integrator import IDAS
from nextpredco.core.model import Model, Plant
from nextpredco.core.optimizer import IPOPT
from nextpredco.core.settings import (
    IDASSettings,
    IPOPTSettings,
    KalmanSettings,
    ModelSettings,
    MPCSettings,
    PIDSettings,
    read_settings_csv,
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
        settings = read_settings_csv()
        tools.print(settings)
        self.builder.set_model(settings['model'], settings['model.integrator'])

        if 'controller' in settings:
            self.builder.set_controller(
                settings['controller'],
                settings['controller.optimizer'],
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
