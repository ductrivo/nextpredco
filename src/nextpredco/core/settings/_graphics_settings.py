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
            est_style = self.est_style.copy()
            est_style['label'] = f'{prefix}{est_style["label"]}'
            return est_style
        if source == 'act':
            act_style = self.act_style.copy()
            act_style['label'] = f'{prefix}{act_style["label"]}'
            return act_style
        if source == 'meas':
            meas_style = self.meas_style.copy()
            meas_style['label'] = f'{prefix}{meas_style["label"]}'
            return meas_style
        if source == 'goal':
            goal_style = self.goal_style.copy()
            goal_style['label'] = f'{prefix}{goal_style["label"]}'
            return goal_style

        msg = f'Unknown source: {source}'
        raise ValueError(msg)
