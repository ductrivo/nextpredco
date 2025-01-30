import importlib.util
from pathlib import Path
from typing import override

import pandas as pd

from nextpredco.core import utils
from nextpredco.core.consts import CONFIG_FOLDER, SETTING_FOLDER
from nextpredco.core.controller import MPC, PID, Controller
from nextpredco.core.custom_types import SourceType
from nextpredco.core.model import Model
from nextpredco.core.observer import Observer
from nextpredco.core.plant import Plant
from nextpredco.core.settings.settings import (
    ControllerSettings,
    IDASSettings,
    KalmanSettings,
    ModelSettings,
    MPCSettings,
    ObserverSettings,
    PIDSettings,
    PlantSettings,
    SettingsFactory,
)


def _cast_value(value: str, value_type: str):
    # if isinstance(value, str):
    #     value = value.lower()
    value_type = value_type.lower()
    if value_type == 'str':
        return str(value)
    if value_type == 'bool':
        return value.lower() == 'true'
    if value_type == 'int':
        return int(value)
    if value_type == 'float':
        return float(value)
    if 'list' in value_type:
        value = (
            value.replace('[', '')
            .replace(']', '')
            .replace("'", '')
            .replace('"', '')
            .replace(' ', '')
        )
        values = value.split(',')
        value_out: list[str | bool | int | float] = []

        if 'str' in value_type:
            value_out = [str(v) for v in values]
        elif 'bool' in value_type:
            value_out = [v.lower() == 'true' for v in values]
        elif 'int' in value_type:
            value_out = [int(v) for v in values]
        elif 'float' in value_type:
            value_out = [float(v) for v in values]

        if len(value_out) == 1 and value_out[0] == '':
            return []
        return value_out
    msg = f'Unsupported type: {value_type}'
    raise ValueError(msg)


def _df_to_nested_dict(df: pd.DataFrame):
    nested_dict: dict = {}
    for _, row in df.iterrows():
        if row['value'] != '':
            keys = row['parameter'].split('.')
            value_type = row['type']

            value = _cast_value(row['value'], value_type)

            d = nested_dict
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d[keys[-1]] = value

    return nested_dict


def extract_settings():
    df = pd.read_csv(
        Path.cwd() / SETTING_FOLDER / 'settings.csv',
        na_filter=False,
    )
    df_dict: dict = _df_to_nested_dict(df)
    utils.pretty_print_dict(df_dict)

    all_settings = {}

    # Integrator settings in model
    if 'integrator' in df_dict['model']:
        all_settings['model_integrator'] = SettingsFactory().create(
            **df_dict['model']['integrator'],
        )
        df_dict['model'].pop('integrator')
    else:
        all_settings['model_integrator'] = None

    all_settings['model'] = SettingsFactory().create(**df_dict['model'])

    if 'controller' in df_dict:
        if 'optimizer' in df_dict['controller']:
            all_settings['controller_optimizer'] = SettingsFactory().create(
                **df_dict['controller']['optimizer'],
            )
            df_dict['controller'].pop('optimizer')
        else:
            all_settings['controller_optimizer'] = None

        all_settings['controller'] = SettingsFactory().create(
            **df_dict['controller'],
        )
    else:
        all_settings['controller'] = None

    if 'observer' in df_dict:
        if 'optimizer' in df_dict['observer']:
            all_settings['observer_optimizer'] = SettingsFactory().create(
                **df_dict['observer']['optimizer'],
            )
            df_dict['observer'].pop('optimizer')
        else:
            all_settings['observer_optimizer'] = None

        all_settings['observer'] = SettingsFactory().create(
            **df_dict['observer'],
        )
    else:
        all_settings['observer'] = None

    if 'plant' in df_dict:
        pass
    else:
        all_settings['plant'] = None

    return all_settings


class ControlSystem:
    def __init__(self):
        self.model: Model | None = None
        self.controller = None
        self.observer = None
        self.plant = None

    def simulate(self):
        pass


class ControlSystemBuilder:
    def __init__(self):
        self.system = ControlSystem()

    def set_model(
        self,
        settings: ModelSettings,
        integrator: IDASSettings,
    ) -> 'ControlSystemBuilder':
        self.system.model = Model(settings, integrator)
        return self

    def set_controller(
        self,
        settings: PIDSettings | MPCSettings,
        optimizer_settings: IDASSettings | None = None,
    ) -> 'ControlSystemBuilder':
        if isinstance(settings, PIDSettings):
            self.system.controller = PID(settings)
        elif isinstance(settings, MPCSettings):
            self.system.controller = MPC(
                settings,
                self.system.model,
                optimizer_settings,
            )
        return self

    def set_observer(self, settings: KalmanSettings) -> 'ControlSystemBuilder':
        # if isinstance(settings, KalmanSettings):
        #     self.system.observer = Observer(settings)
        return self

    def set_plant(self, plant: Plant) -> 'ControlSystemBuilder':
        self.system.plant = plant
        return self

    def build(self) -> ControlSystem:
        return self.system


class Director:
    def __init__(self, builder: ControlSystemBuilder):
        self.builder = builder

    def construct(self):
        settings = extract_settings()
        utils.print(settings)
        self.builder.set_model(settings['model'], settings['model_integrator'])

        if 'controller' in settings:
            self.builder.set_controller(
                settings['controller'],
                settings['controller_optimizer'],
            )

        if 'observer' in settings:
            self.builder.set_observer(settings['observer'])

        if 'plant' in settings:
            self.builder.set_plant(settings['plant'])
        return self.builder.build()


if __name__ == '__main__':
    builder = ControlSystemBuilder()
    director = Director(builder)
    system = director.construct()
