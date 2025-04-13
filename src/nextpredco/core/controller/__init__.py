from nextpredco.core.controller._controller2 import (
    ControllerABC as ControllerABC,
)
from nextpredco.core.controller._mpc2 import MPC as MPC
from nextpredco.core.controller._pid import PID as PID
from nextpredco.core.model import ModelABC
from nextpredco.core.settings import (
    ControllerSettings,
    IntegratorSettings,
    MPCSettings,
    OptimizerSettings,
    PIDSettings,
)

type Controller = PID | MPC
__all__ = ['MPC', 'PID', 'Controller', 'ControllerFactory']


class ControllerFactory:
    @staticmethod
    def create(
        settings: ControllerSettings,
        model: ModelABC | None = None,
        optimizer_settings: OptimizerSettings | None = None,
        integrator_settings: IntegratorSettings | None = None,
    ):
        if isinstance(settings, PIDSettings):
            return PID(settings)

        if isinstance(settings, MPCSettings):
            if model is None:
                msg = 'ModelABC is required for MPC controller.'
                raise ValueError(msg)

            if optimizer_settings is None:
                msg = 'Optimizer settings are required for MPC controller.'
                raise ValueError(msg)

            return MPC(
                settings=settings,
                model=model,
                optimizer_settings=optimizer_settings,
                integrator_settings=integrator_settings,
            )
        raise NotImplementedError
