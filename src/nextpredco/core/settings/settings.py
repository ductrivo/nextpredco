import itertools
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType

import pandas as pd

from nextpredco.core import utils
from nextpredco.core.consts import (
    CONFIG_FOLDER,
    SETTING_FOLDER,
    SS_VARS_PRIMARY,
    SS_VARS_SECONDARY,
)
from nextpredco.core.errors import SystemVariableError
from nextpredco.core.logger import logger


@dataclass
class ModelSettings:
    name: str = field(default='model')
    is_continuous: bool = field(default=True)
    is_linear: bool = field(default=False)

    k: int = field(default=0)
    k_clock: int = field(default=0)

    dt: float = field(default=1)
    dt_clock: float = field(default=1)

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
    # model_info_path: Path = field(
    #     default=Path.cwd() / 'settings' / 'model_info.csv',
    # )


@dataclass(kw_only=True)
class ControllerSettings:
    name: str = 'controller'
    dt: float = field(default=1.0)


@dataclass
class OptimizerSettings:
    name: str = field(default='optimizer')


@dataclass
class IPOPTSettings(OptimizerSettings):
    name: str = field(default='ipopt')
    opts: dict[str, str | float | int] = field(default_factory=dict)


@dataclass(kw_only=True)
class PIDSettings(ControllerSettings):
    kp: float = field(default=1.0)
    ki: float = field(default=0.0)
    kd: float = field(default=0.0)


@dataclass(kw_only=True)
class MPCSettings(ControllerSettings):
    name: str = field(default='mpc')
    n_pred: int = field(default=1)
    # optimizer: Optimizer | None = field(default=None)


@dataclass(kw_only=True)
class IntegratorSettings:
    name: str = field(default='integrator')
    # n_pred: int = field(default=1)
    # optimizer: Optimizer | None = field(default=None)


@dataclass(kw_only=True)
class IDASSettings(IntegratorSettings):
    name: str = field(default='idas')
    opts: dict[str, str | float | int] = field(default_factory=dict)
    # _opts_types: dict[str, str] = field(default_factory=dict)
    # n_pred: int = field(default=1)
    # optimizer: Optimizer | None = field(default=None)


@dataclass
class ObserverSettings:
    name: str = field(default='observer')


@dataclass
class KalmanSettings(ObserverSettings):
    pass


@dataclass
class PlantSettings:
    name: str = field(default='plant')


class SettingsFactory:
    @staticmethod
    def create(name, **settings):
        # Model
        if name == 'model':
            return ModelSettings(**settings)

        # Model's integrator
        if name == 'idas':
            return IDASSettings(**settings)

        # Controllers
        if name == 'pid':
            return PIDSettings(**settings)

        if name == 'mpc':
            return MPCSettings(**settings)

        if name == 'ipopt':
            return IPOPTSettings(**settings)

        msg = f'Unknown settings type: {name}'
        raise ValueError(msg)


def create_settings_template(project_dir: Path | None = None):
    # Set the project directory
    if project_dir is None:
        project_dir = Path.cwd()

    # Get the configuration and equations file paths
    config_dir = project_dir / CONFIG_FOLDER
    equations_file = config_dir / 'equations.py'
    config_file = config_dir / 'config.toml'

    # Check if settings/equations.py and settings/model_info.csv exists
    if not equations_file.exists() or not config_file.exists():
        logger.error(
            'Settings files not found. Please initialize a project first. '
            'Exiting...',
        )
        return

    # Load the configuration file
    with config_file.open('rb') as f:
        configs = tomllib.load(f)

    # Create the DataFrames
    settings_dfs: list[pd.DataFrame] = []

    # Create the model settings DataFrame
    settings_dfs.append(
        _create_settings_df(ModelSettings(), prefix='model'),
    )

    # Create the MODEL.INFO DataFrame
    settings_dfs.append(_create_model_info_df(configs))

    # Create the INTEGRATOR settings DataFrame
    if configs['system']['model']['is_continuous']:
        integrator_name = configs['system']['model']['integrator']['name']
        settings_dfs.append(
            _create_settings_df(
                SettingsFactory.create(name=integrator_name),
                prefix='model.integrator',
            ),
        )

    # Create the CONTROLLER settings DataFrame
    name = configs['system']['controller']['name']
    settings_dfs.append(
        _create_settings_df(
            SettingsFactory.create(name=name),
            prefix='controller',
        ),
    )

    # Create the OPTIMIZER settings DataFrame
    settings_dfs.append(
        _create_settings_df(
            SettingsFactory.create(
                name=configs['system']['controller']['optimizer']['name'],
            ),
            prefix='controller.optimizer',
        ),
    )

    # Concatenate the DataFrames
    settings_df = pd.concat(settings_dfs, ignore_index=True)

    # Create the settings directory
    settings_dir = project_dir / SETTING_FOLDER
    settings_dir.mkdir(exist_ok=True)

    # Create the settings file path
    df_path = settings_dir / 'settings_template.csv'

    # Convert the list values to strings
    # If not do this, list[str] such as ['x1', 'x2']
    # will be converted to "['x1', 'x2']".
    # With this, we have "[x1, x2]" which is more readable.
    settings_df['value'] = [
        utils.list_to_str(value) if isinstance(value, list) else value
        for value in settings_df['value']
    ]

    # Lowercase the string values
    settings_df = settings_df.map(
        lambda x: x.lower() if isinstance(x, str) else x,
    )

    # Save the DataFrame to a CSV file
    settings_df.to_csv(df_path, index=False)
    logger.debug('Settings file created successfully at %s.', df_path)


def _create_model_info_df(
    configs: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> pd.DataFrame:
    df: dict[str, list] = {
        'parameter': [],
        'type': [],
        'value': [],
    }

    # Add x, z, u, p, q_vars to df
    # vars_: is used for validating variable definitions
    first_vars: dict[str, list[str]] = {}
    for ss_var in SS_VARS_PRIMARY:
        vars_list = list(configs['state_space'][ss_var].keys())
        df['parameter'].append(f'model.{ss_var}_vars')
        df['type'].append('list[str]')
        df['value'].append(vars_list)
        first_vars[ss_var] = vars_list

    # Add m, o, y_vars to df
    second_vars: dict[str, list[str]] = {}
    for ss_var in SS_VARS_SECONDARY:
        vars_list = list(configs['state_space']['x'].keys())
        df['parameter'].append(f'model.{ss_var}_vars')
        df['type'].append('list[str]')
        df['value'].append(vars_list)
        second_vars[ss_var] = vars_list

    # Add upq_vars
    upq_vars = first_vars['u'] + first_vars['p'] + first_vars['q']
    df['parameter'].append('model.upq_vars')
    df['type'].append('list[str]')
    df['value'].append(upq_vars)

    # Add physical parameters to df
    all_vars: list[str] = []
    for ss_var in SS_VARS_PRIMARY:
        for key, info in configs['state_space'][ss_var].items():
            df['parameter'].append(f'model.info.{key}')
            df['type'].append('float')
            df['value'].append(info['value'])
            all_vars.append(key)

    # All vars
    # TODO: move to model
    _validate_definitions(
        first_vars=first_vars,
        second_vars=second_vars,
        all_vars=all_vars,
    )
    logger.debug('Validated system variable declarations.')
    return pd.DataFrame(df)


def _create_settings_df(
    settings: ModelSettings | IDASSettings | MPCSettings | OptimizerSettings,
    prefix='',
) -> pd.DataFrame:
    data: dict[str, list] = {
        'parameter': [],
        'type': [],
        'value': [],
    }
    df_opts = pd.DataFrame()
    for field_ in fields(settings):
        if field_.name == 'opts':
            # Get options from file
            df_opts = _get_settings_from_file(settings.name)
            df_opts = df_opts[['parameter', 'type', 'value']]
            df_opts.loc[:, 'parameter'] = df_opts['parameter'].apply(
                lambda x: f'{prefix}.opts.{x}',
            )
        elif (
            field_.name not in ['info']
            and '_types' not in field_.name
            and '_vars' not in field_.name
        ):
            # Get the type of the field
            if isinstance(field_.type, UnionType):
                data['parameter'].append(f'{prefix}.{field_.name}')
                data['type'].append(str(field_.type))
                data['value'].append(getattr(settings, field_.name))

            elif field_.type.__name__ == 'list':
                if isinstance(field_.type.__args__[0], UnionType):
                    type_0 = str(field_.type.__args__[0])
                else:
                    type_0 = field_.type.__args__[0].__name__

                data['parameter'].append(f'{prefix}.{field_.name}')
                data['type'].append(f'list[{type_0}]')
                data['value'].append(getattr(settings, field_.name))

            elif field_.type.__name__ == 'dict':
                values_dict = getattr(settings, field_.name)
                types_dict = getattr(settings, f'_{field_.name}_types')

                for key, val in values_dict.items():
                    data['parameter'].append(f'{prefix}.{field_.name}.{key}')
                    data['type'].append(types_dict[key])
                    data['value'].append(val)

            else:
                data['parameter'].append(f'{prefix}.{field_.name}')
                data['type'].append(field_.type.__name__)
                data['value'].append(getattr(settings, field_.name))

    df = pd.DataFrame(data)

    # Concatenate the DataFrames
    return pd.concat([df, df_opts], ignore_index=True)


def extract_settings_from_file(
    file_name: str = 'settings_template.csv',
) -> dict:
    # Read the settings file
    df = pd.read_csv(
        Path.cwd() / SETTING_FOLDER / file_name,
        na_filter=False,
    )
    input(df)

    # Convert the DataFrame to a nested dictionary
    # TODO: add type hints
    df_dict = _df_to_nested_dict(df)

    logger.debug(df_dict)
    input('Press Enter to continue...')
    all_ = {}

    # Get INTEGRATOR settings in model
    if 'integrator' in df_dict['model']:
        all_['model_integrator'] = SettingsFactory().create(
            **df_dict['model']['integrator'],
        )
        df_dict['model'].pop('integrator')
    else:
        all_['model_integrator'] = None

    # Get MODEL settings
    all_['model'] = SettingsFactory().create(**df_dict['model'])

    # Get CONTROLLER and OPTIMIZER settings
    if 'controller' in df_dict:
        if 'optimizer' in df_dict['controller']:
            all_['controller_optimizer'] = SettingsFactory().create(
                **df_dict['controller']['optimizer'],
            )
            df_dict['controller'].pop('optimizer')
        else:
            all_['controller_optimizer'] = None

        # Get CONTROLLER settings
        all_['controller'] = SettingsFactory().create(**df_dict['controller'])
    else:
        all_['controller'] = None

    # Get OBSERVER and OPTIMIZER settings
    if 'observer' in df_dict:
        # Get OPTIMIZER settings
        if 'optimizer' in df_dict['observer']:
            all_['observer_optimizer'] = SettingsFactory().create(
                **df_dict['observer']['optimizer'],
            )
            df_dict['observer'].pop('optimizer')
        else:
            all_['observer_optimizer'] = None

        # Get OBSERVER settings
        all_['observer'] = SettingsFactory().create(**df_dict['observer'])
    else:
        all_['observer'] = None

    # Get PLANT settings
    if 'plant' in df_dict:
        pass
    else:
        all_['plant'] = None

    return all_


def _validate_definitions(
    first_vars: dict[str, list[str]],
    second_vars: dict[str, list[str]],
    all_vars: list[str],
):
    primary_vars = list(itertools.chain(*first_vars.values()))

    if (len(primary_vars) != len(all_vars)) and (
        set(primary_vars) != set(all_vars)
    ):
        msg = (
            'The total length of x/z/u/p/q_vars must be equal to '
            'the number of variables declared in model.info.'
        )
        raise ValueError(msg)

    # ss_var: state space variable
    for ss_var, vars_list in second_vars.items():
        if not set(vars_list).issubset(set(all_vars)):
            raise SystemVariableError(ss_var, 'physical variables')


def _get_settings_from_file(name: str) -> pd.DataFrame:
    setting_file = Path(__file__).parent / f'{name}_options.csv'
    return pd.read_csv(setting_file, na_filter=False)


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


def _df_to_nested_dict(df: pd.DataFrame) -> dict:
    nested_dict: dict = {}
    for _, row in df.iterrows():
        if row['value'] != '':
            keys = row['parameter'].split('.', 3)
            value_type = row['type']
            value = _cast_value(row['value'], value_type)

            d = nested_dict
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d[keys[-1]] = value
    return nested_dict
