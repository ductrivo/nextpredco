from typing import override

from nextpredco.core.controller._controller import ControllerABC
from nextpredco.core.settings import (
    PIDSettings,
)


class PID(ControllerABC):
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
