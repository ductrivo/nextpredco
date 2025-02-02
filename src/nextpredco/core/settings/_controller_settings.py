from dataclasses import dataclass, field


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
    n_pred: int = field(default=1)
