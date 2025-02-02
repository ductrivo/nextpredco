from dataclasses import dataclass, field

PARAMETER = 'parameter'
TYPE = 'type'
VALUE = 'value'
TEX = 'tex'
DESCRIPTION = 'description'
ROLE = 'role'


@dataclass
class OptimizerSettings:
    name: str = field(default='optimizer')
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class IPOPTSettings(OptimizerSettings):
    name: str = field(default='ipopt')
    opts: dict[str, str | float | int] = field(default_factory=dict)
