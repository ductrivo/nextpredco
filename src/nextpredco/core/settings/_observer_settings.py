from dataclasses import dataclass, field

type ObserverSettings = KalmanSettings


@dataclass
class ObserverSettingsAbstract:
    name: str = field(default='observer')
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class KalmanSettings(ObserverSettingsAbstract):
    pass
