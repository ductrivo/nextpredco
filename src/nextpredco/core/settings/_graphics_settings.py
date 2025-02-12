from dataclasses import dataclass, field

type MatplotlibPlotStyle = dict[str, str | float | tuple]


@dataclass
class GraphicsSettings:
    est_style: MatplotlibPlotStyle = field(
        default_factory=lambda: {
            'label': 'Estimated',
            'color': 'blue',
            'linewidth': 1,
        }
    )

    pred_style: MatplotlibPlotStyle = field(
        default_factory=lambda: {
            'label': 'Predictions',
            'color': 'red',
            'linewidth': 1,
            'linestyle': ':',
        }
    )

    fine_pred_style: MatplotlibPlotStyle = field(
        default_factory=lambda: {
            'label': 'Fine predictions',
            'color': 'green',
            'linewidth': 1,
        }
    )

    act_style: MatplotlibPlotStyle = field(
        default_factory=lambda: {
            'label': 'Actual',
            'color': 'red',
            'linestyle': ':',
            'linewidth': 1,
        }
    )

    meas_style: MatplotlibPlotStyle = field(
        default_factory=lambda: {
            'label': 'Measured',
            'color': 'lime',
            'linestyle': 'dotted',
            'alpha': 0.9,
        }
    )

    goal_style: MatplotlibPlotStyle = field(
        default_factory=lambda: {
            'color': 'black',
            'linewidth': 0.5,
            'linestyle': (0, (5, 5)),
        }
    )

    def get_style(self, source: str, prefix: str = '') -> MatplotlibPlotStyle:
        if source == 'est':
            style = self.est_style.copy()
            style['label'] = f'{prefix}{style["label"]}'
            return style
        if source == 'pred':
            style = self.pred_style.copy()
            style['label'] = f'{prefix}{style["label"]}'
            return style
        if source == 'fine_pred':
            style = self.fine_pred_style.copy()
            style['label'] = f'{prefix}{style["label"]}'
            return style

        if source == 'act':
            style = self.act_style.copy()
            style['label'] = f'{prefix}{style["label"]}'
            return style
        if source == 'meas':
            style = self.meas_style.copy()
            style['label'] = f'{prefix}{style["label"]}'
            return style
        if source == 'goal':
            style = self.goal_style.copy()
            style['label'] = f'{prefix}{style["label"]}'
            return style

        msg = f'Unknown source: {source}'
        raise ValueError(msg)
