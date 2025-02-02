from dataclasses import dataclass, field


@dataclass(kw_only=True)
class IntegratorSettings:
    name: str = field(default='integrator')
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass(kw_only=True)
class IDASSettings(IntegratorSettings):
    name: str = field(default='idas')
    opts: dict[str, str | float | int] = field(default_factory=dict)
