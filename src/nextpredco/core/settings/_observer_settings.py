from dataclasses import dataclass, field

type ObserverSettings = KalmanSettings | StateFeedback


@dataclass
class ObserverSettingsAbstract:
    name: str = field(default='observer')
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class StateFeedback(ObserverSettingsAbstract):
    name: str = field(default='state_feedback')


@dataclass
class KalmanSettings(ObserverSettingsAbstract):
    name: str = field(default='kalman')
