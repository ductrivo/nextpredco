from dataclasses import dataclass, field

type MatplotlibPlotStyle = dict[str, str | float | tuple]


@dataclass
class GraphicsSettings:
    est_style: MatplotlibPlotStyle = field(
        default_factory=lambda: {
            'label': 'Estimated',
            # 'color': 'blue',
            'linewidth': 1,
        }
    )

    act_style: MatplotlibPlotStyle = field(
        default_factory=lambda: {
            'label': 'Actual',
            # 'color': 'red',
            'linestyle': ':',
            'linewidth': 1,
        }
    )

    meas_style: MatplotlibPlotStyle = field(
        default_factory=lambda: {
            'label': 'Measured',
            # 'color': 'lime',
            'linestyle': 'dotted',
            'alpha': 0.9,
        }
    )

    goal_style: MatplotlibPlotStyle = field(
        default_factory=lambda: {
            # 'color': 'black',
            'linewidth': 0.5,
            'linestyle': (0, (5, 5)),
        }
    )

    def get_style(self, source: str, prefix: str = '') -> MatplotlibPlotStyle:
        if source == 'est':
            self.est_style['label'] = f'{prefix}{self.est_style["label"]}'
            return self.est_style
        if source == 'act':
            self.act_style['label'] = f'{prefix}{self.act_style["label"]}'
            return self.act_style
        if source == 'meas':
            self.meas_style['label'] = f'{prefix}{self.meas_style["label"]}'
            return self.meas_style
        if source == 'goal':
            self.goal_style['label'] = f'{prefix}{self.goal_style["label"]}'
            return self.goal_style

        msg = f'Unknown source: {source}'
        raise ValueError(msg)
