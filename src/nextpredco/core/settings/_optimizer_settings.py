from dataclasses import dataclass, field

type OptimizerSettings = IPOPTSettings


@dataclass
class OptimizerSettingsAbstract:
    name: str = field(default='optimizer')
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class IPOPTSettings(OptimizerSettingsAbstract):
    name: str = field(default='ipopt')
    opts: dict[str, str | float | int] = field(default_factory=dict)
