from dataclasses import dataclass, field

PARAMETER = 'parameter'
TYPE = 'type'
VALUE = 'value'
TEX = 'tex'
DESCRIPTION = 'description'
ROLE = 'role'


@dataclass
class PlantSettings:
    name: str = field(default='plant')
    descriptions: dict[str, str] = field(default_factory=dict)
