from dataclasses import dataclass, field


@dataclass
class ModelSettings:
    name: str = field(default='model')
    is_continuous: bool = field(default=True)
    is_linear: bool = field(default=False)

    k: int = field(default=0)
    k_clock: int = field(default=0)

    dt: float = field(default=0.01)
    dt_clock: float = field(default=0.01)

    t_max: float = field(default=10.0)

    sources: list[str] = field(
        default_factory=lambda: [
            'goal',
            'est',
            'act',
            'meas',
            'filt',
            'meas_clock',
            'filt_clock',
        ],
    )

    info: dict[str, float] = field(default_factory=dict)
    x_vars: list[str] = field(default_factory=list)
    z_vars: list[str] = field(default_factory=list)
    u_vars: list[str] = field(default_factory=list)
    p_vars: list[str] = field(default_factory=list)
    q_vars: list[str] = field(default_factory=list)
    m_vars: list[str] = field(default_factory=list)
    o_vars: list[str] = field(default_factory=list)
    y_vars: list[str] = field(default_factory=list)
    upq_vars: list[str] = field(default_factory=list)
    const_vars: list[str] = field(default_factory=list)

    tex: dict[str, str] = field(default_factory=dict)
    descriptions: dict[str, str] = field(
        default_factory=lambda: {
            'is_continuous': (
                'True if the model is continuous, '
                'False if the model is discrete.'
            ),
            'is_linear': (
                'True if the model is linear, '
                'False if the model is non-linear.'
            ),
            'k': 'Current time step.',
            'k_clock': 'Current time step for the clock.',
            'dt': 'Discretization time step for the model.',
            'dt_clock': 'Discretization time step for the clock.',
            't_max': (
                'Maximum operation time. '
                'This will be used for memory allocation.'
            ),
            'sources': (
                'List of sources for the model. '
                'Possible values: goal, est, act, meas, filt, '
                'meas_clock, filt_clock.'
            ),
            'info': 'Physical parameters of the model.',
            'x_vars': 'Physical parameters declared as State variables.',
            'z_vars': 'Physical parameters declared as Algebraic variables.',
            'u_vars': (
                'Physical parameters declared as Control Input variables.'
            ),
            'p_vars': 'Physical parameters declared as Disturbance variables.',
            'q_vars': (
                'Physical parameters declared as Known Varying variables.'
            ),
            'upq_vars': 'Physical parameters declared as u/p/q_vars.',
            'y_vars': 'Physical parameters declared as Controlled variables.',
            'm_vars': 'Physical parameters declared as Measured variables.',
            'o_vars': 'Measured variables that are used in Observer.',
        }
    )
