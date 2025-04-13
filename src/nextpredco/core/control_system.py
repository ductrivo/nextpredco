from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nextpredco.core import tools
from nextpredco.core._logger import logger
from nextpredco.core._sync import GlobalState
from nextpredco.core._typing import TgridType
from nextpredco.core.controller import (
    Controller,
    ControllerFactory,
)
from nextpredco.core.model import ModelABC, Plant
from nextpredco.core.settings import (
    ControllerSettings,
    IntegratorSettings,
    ModelSettings,
    ObserverSettings,
    OptimizerSettings,
    get_settings,
)


class ControlSystem:
    def __init__(self):
        self.model: ModelABC | None = None  # type: ignore[annotation-unchecked]
        self.controller: Controller | None = None  # type: ignore[annotation-unchecked]
        self.observer = None
        self.plant: Plant | None = None
        self.filter = None

        structure = {
            'plant': {
                'element': self.plant,
                'input': self.controller,
                'output': self.controller,
            },
        }

    def simulate_model_only(
        self,
        t_grid: TgridType,
        x_arr: NDArray,
        u_arr: NDArray | None = None,
        p_arr: NDArray | None = None,
        q_arr: NDArray | None = None,
    ):
        if u_arr is not None:
            k_max = u_arr.shape[1]
        elif p_arr is not None:
            k_max = p_arr.shape[1]
        elif q_arr is not None:
            k_max = q_arr.shape[1]
        else:
            msg = 'At least one of u_arr, p_arr, or q_arr must be provided'
            raise ValueError(msg)

        x = x_arr[:, 0, None]
        x_results: list[NDArray] = [x]
        for k in range(k_max):
            u = None if u_arr is None else u_arr[:, k, None]
            p = None if p_arr is None else p_arr[:, k, None]
            q = None if q_arr is None else q_arr[:, k, None]

            while True:
                logger.debug(
                    f'k={k}, t={self.plant.t.vec[0, 0]}, t_grid = {t_grid}\nu={u.T}, p={p.T}, q={q}'
                )
                self.plant.make_step(u=u, p=p, q=q)
                if tools.is_in_list(t=self.plant.t.vec[0, 0], t_grid=t_grid):
                    break

            x_results.append(x)
        x_results_ = np.hstack(x_results)
        return x_results_, None

    def simulate2(self):
        logger.info(
            'Total size of ModelData: %s MB.', self.model._data.size / 1024
        )
        input('Press Enter to start simulation')

        for k in range(self.model.k_max):
            u = self.controller.make_step()
            self.plant.make_step(u=u)

        self.plant.export_data_csv()

    def simulate(self):
        structure = OrderedDict()

        #
        structure[self.controller] = [
            ['x', self.plant, 'm'],
            ['u', self.plant, 'u'],
            ['p', self.plant, 'p'],
            ['q', self.plant, 'q'],
        ]
        structure[self.plant] = [['u', self.controller, 'u']]

        for k in range(GlobalState.k_max()):
            for element, in_outs in structure.items():
                self.controller.goals.y.arr = np.array([[0.8]])

                self.get_inputs(element, in_outs)
                element.make_step()

        self.export_data_csv()
        return self

    def export_data_csv(self, k0: int | None = None, kf: int | None = None):
        k0 = k0 if k0 is not None else 0
        kf = kf if kf is not None else GlobalState.k()

        report_dir = Path().cwd() / 'report'
        report_dir.mkdir(exist_ok=True)

        dfs_: list[pd.DataFrame] = [
            pd.DataFrame(np.arange(k0, kf + 1), columns=['k']),
            self.plant.t.df,
            self.plant.p.df,
            self.plant.q.df,
            self.plant.x.df,
            self.plant.u.df,
        ]
        dfs = pd.concat(dfs_, axis=1)

        # input(f'dfs = {dfs}')
        # input(f'df_cost = {self.controller.costs.df}')
        dfs = dfs.merge(self.controller.goals.df, how='left', on='k')
        dfs = dfs.merge(self.controller.costs.df, how='left', on='k')

        dfs.to_csv(
            report_dir / 'data.csv',
            index=False,
            float_format='%.3f',
        )

        self.controller.predictions.df.to_csv(
            report_dir / 'predictions.csv',
            index=False,
            float_format='%.3f',
        )

    @staticmethod
    def get_inputs(dest, in_outs):
        for input_key, source, output_key in in_outs:
            # logger.debug(
            #     f'Source: {source}\n'
            #     f'Input Key: {output_key}\n'
            #     f'Dest: {dest}\n'
            #     f'Output Key: {input_key}\n'
            # )
            source_output = getattr(*source.outputs[output_key])
            setattr(*dest.inputs[input_key], source_output)

            # input('Check get inputs')


class ControlSystemBuilder:
    def __init__(self):
        self.system = ControlSystem()

    def set_plant(
        self,
        settings: ModelSettings,
        integrator_settings: IntegratorSettings | None = None,
    ) -> 'ControlSystemBuilder':
        # Create model
        self.system.plant = Plant(settings, integrator_settings)
        return self

    def set_controller(
        self,
        settings: ControllerSettings,
        optimizer_settings: OptimizerSettings | None = None,
        integrator_settings: IntegratorSettings | None = None,
    ) -> 'ControlSystemBuilder':
        # Create controller based on the setting type
        self.system.controller = ControllerFactory.create(
            settings=settings,
            model=self.system.plant,  # TODO: which model?
            optimizer_settings=optimizer_settings,
            integrator_settings=integrator_settings,
        )

        return self

    def set_observer(
        self,
        settings: ObserverSettings,
        optimizer_settings: OptimizerSettings | None = None,
        integrator_settings: IntegratorSettings | None = None,
    ) -> 'ControlSystemBuilder':
        return self

    # def set_plant(self, plant: Plant) -> 'ControlSystemBuilder':
    #     self.system.plant = plant
    #     return self

    def build(self) -> ControlSystem:
        return self.system


class Director:
    def __init__(self, builder: ControlSystemBuilder):
        self.builder = builder

    def construct(
        self,
        settings: dict,
    ) -> ControlSystem:
        # Construct Plant
        self.builder.set_plant(settings['model'], settings['model.integrator'])

        # Construct Controller
        # TODO: Where to construct the model?
        if 'controller' in settings:
            self.builder.set_controller(
                settings=settings['controller'],
                optimizer_settings=settings['controller.optimizer'],
                integrator_settings=settings['controller.integrator'],
            )

        if 'observer' in settings:
            self.builder.set_observer(
                settings=settings['observer'],
                optimizer_settings=settings['observer.optimizer'],
                integrator_settings=settings['observer.integrator'],
            )

        # if 'plant' in settings:
        #     self.builder.set_plant(settings['plant'])
        return self.builder.build()


def construct_control_system(
    setting_file_name: str = 'settings_template.csv',
) -> ControlSystem:
    # To construct a control system,
    # we need to read the settings from a CSV file
    settings = get_settings(setting_file_name)

    builder = ControlSystemBuilder()
    director = Director(builder)
    return director.construct(settings)
