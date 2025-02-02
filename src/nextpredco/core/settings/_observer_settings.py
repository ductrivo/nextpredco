from dataclasses import dataclass, field


@dataclass
class ObserverSettings:
    name: str = field(default='observer')
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class KalmanSettings(ObserverSettings):
    pass
