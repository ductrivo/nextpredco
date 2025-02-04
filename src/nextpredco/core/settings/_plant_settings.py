from dataclasses import dataclass, field


@dataclass
class PlantSettings:
    name: str = field(default='plant')
    descriptions: dict[str, str] = field(default_factory=dict)
