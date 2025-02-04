from dataclasses import dataclass, field

type IntegratorSettings = IDASSettings | TaylorSettings


@dataclass(kw_only=True)
class IntegratorSettingsAbstract:
    name: str = field(default='integrator')
    descriptions: dict[str, str] = field(default_factory=dict)
    h: float | None = field(default=None)


@dataclass
class TaylorSettings:
    name: str = field(default='taylor')
    order: int = field(default=1)
    h: float = field(default=-1)
    opts: dict = field(default_factory=dict)

    tex: dict[str, str] = field(
        default_factory=lambda: {
            'order': 'S',
            'h': 'h',
        }
    )

    descriptions: dict[str, str] = field(
        default_factory=lambda: {
            'name': 'Integrator name.',
            'order': 'Taylor series order.',
            'h': 'Step size.',
            'opts': 'Additional options.',  # TODO: any?
        }
    )


@dataclass(kw_only=True)
class IDASSettings(IntegratorSettingsAbstract):
    name: str = field(default='idas')
    opts: dict[str, str | float | int] = field(default_factory=dict)
    tex: dict[str, str] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
