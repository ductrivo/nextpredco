from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(kw_only=True)
class ControllerSettings:
    name: str = 'controller'
    dt: float = field(default=-1)
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass(kw_only=True)
class PIDSettings(ControllerSettings):
    kp: float = field(default=1.0)
    ki: float = field(default=0.0)
    kd: float = field(default=0.0)


@dataclass(kw_only=True)
class MPCSettings(ControllerSettings):
    name: str = field(default='mpc')
    n_pred: int = field(default=3)
    normalizing: bool = field(default=False)

    weight_x: list[float] = field(default_factory=lambda: [1.0])
    weight_y: list[float] = field(default_factory=lambda: [1.0])
    weight_u: list[float] = field(default_factory=lambda: [1.0])
    weight_du: list[float] = field(default_factory=lambda: [1.0])
