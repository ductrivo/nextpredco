import itertools
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


@dataclass(kw_only=True)
class ControllerSettings:
    name: str = 'controller'
    dt: float = field(default=-1)
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class OptimizerSettings:
    name: str = field(default='optimizer')
    descriptions: dict[str, str] = field(default_factory=dict)


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


@dataclass(kw_only=True)
class IntegratorSettings:
    name: str = field(default='integrator')
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass(kw_only=True)
class IDASSettings(IntegratorSettings):
    name: str = field(default='idas')
    opts: dict[str, str | float | int] = field(default_factory=dict)


@dataclass
class ObserverSettings:
    name: str = field(default='observer')
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class KalmanSettings(ObserverSettings):
    pass


@dataclass
class PlantSettings:
    name: str = field(default='plant')
    descriptions: dict[str, str] = field(default_factory=dict)


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
    # TODO: Check if declared variables are consistent with model.info

    # Set the project directory
    if project_dir is None:
        project_dir = Path.cwd()

    # Get the configuration and equations file paths
    config_dir = project_dir / CONFIG_FOLDER
    equations_file = config_dir / 'equations.py'
    config_file = config_dir / 'config.csv'
    info_file = config_dir / 'model_info.csv'

    # Check if settings/equations.py and settings/model_info.csv exists
    if not equations_file.exists() or not config_file.exists():
        logger.error(
            'Settings files not found. Please initialize a project first. '
            'Exiting...',
        )
        return

    # Read the model info file
    df_info, df_vars = _read_model_info_csv(info_file)

    # Create the model settings DataFrame
    df_model = _get_class_settings(prefix='model')

    # Update the model variables
    df_model = _update_model_vars(df=df_model, df_vars=df_vars)

    # Create the DataFrames
    settings_dfs: list[pd.DataFrame] = [df_model, df_info]

    # Read the configuration file
    df_configs = _read_config_csv(config_file)

    # Create the INTEGRATOR settings DataFrame
    if _get_value(df=df_configs, parameter='model.is_continuous'):
        settings_dfs.append(
            _get_class_settings(df=df_configs, prefix='model.integrator'),
        )

    # Create the CONTROLLER settings DataFrame
    settings_dfs.append(
        _get_class_settings(df=df_configs, prefix='controller'),
    )

    # Create the OPTIMIZER settings DataFrame
    settings_dfs.append(
        _get_class_settings(df=df_configs, prefix='controller.optimizer'),
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
    settings_df[['parameter', 'type', 'value']] = settings_df[
        ['parameter', 'type', 'value']
    ].map(
        lambda x: x.lower() if isinstance(x, str) else x,
    )

    # Save the DataFrame to a CSV file
    settings_df.to_csv(df_path, index=False)
    logger.debug('Settings file created successfully at %s.', df_path)


def _read_config_csv(config_file: Path) -> pd.DataFrame:
    return pd.read_csv(config_file, na_filter=False)


def _read_model_info_csv(
    info_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(info_file, na_filter=False)
    data: dict[str, list] = {
        'parameter': [],
        'type': [],
        'value': [],
        'tex': [],
        'description': [],
    }
    data_vars: dict[str, list[str]] = {
        'x': [],
        'z': [],
        'u': [],
        'p': [],
        'q': [],
        'm': [],
        'o': [],
        'y': [],
        'upq': [],
        'const': [],
    }
    for _, row in df.iterrows():
        if row['value'] != '':
            data['parameter'].append('model.info.' + row['parameter'])
            data['type'].append(row['type'])
            data['value'].append(_cast_value(row['value'], row['type']))
            data['description'].append(row['description'])
            data['tex'].append(row['tex'])

            for ss_var in SS_VARS_PRIMARY:
                if ss_var in row['role'] and row['role'] != 'const':
                    data_vars[ss_var].append(row['parameter'])

            for ss_var in SS_VARS_SECONDARY:
                if ss_var in row['role'] and row['role'] != 'const':
                    data_vars[ss_var].append(row['parameter'])

            if row['role'] == 'const':
                data_vars['const'].append(row['parameter'])

    data_vars['upq'] = data_vars['u'] + data_vars['p'] + data_vars['q']

    df_vars: dict[str, list] = {
        'parameter': [],
        'value': [],
    }
    for key, value in data_vars.items():
        df_vars['parameter'].append(f'model.{key}_vars')
        df_vars['value'].append(value)

    return pd.DataFrame(data), pd.DataFrame(df_vars)


def _update_model_vars(
    df: pd.DataFrame, df_vars: pd.DataFrame
) -> pd.DataFrame:
    # Merge df with df_vars on 'parameter' column
    merged_df = df.merge(
        df_vars,
        on='parameter',
        suffixes=('', '_new'),
        how='left',
    )

    # Update the 'value' column in df with the 'value_new' column from df_vars
    merged_df['value'] = merged_df['value_new'].combine_first(
        merged_df['value']
    )

    # Drop the 'value_new' column
    return merged_df.drop(columns=['value_new'])


def _get_type(df: pd.DataFrame, parameter: str) -> str:
    return df.loc[df['parameter'] == parameter, 'type'].values[0]


def _get_value(df: pd.DataFrame, parameter: str) -> str:
    value = df.loc[df['parameter'] == parameter, 'value'].values[0]
    type_ = _get_type(df, parameter)
    return _cast_value(value=value, value_type=type_)


def _get_class_settings(
    df: pd.DataFrame | None = None, prefix: str = ''
) -> pd.DataFrame:
    if df is None and prefix == 'model':
        settings = SettingsFactory.create(name='model')
    elif df is not None:
        try:
            settings = SettingsFactory.create(
                name=_get_value(df=df, parameter=prefix + '.name'),
            )
        except ValueError:
            # TODO: check if this is the best way to handle this
            msg = f'Unknown element name with prefix = {prefix}.'
            raise ValueError(msg) from None

    # Create a dictionary to store the data
    data: dict[str, list] = {
        'parameter': [],
        'type': [],
        'value': [],
        'tex': [],
        'description': [],
    }

    # Create an empty DataFrame for the options
    # in case the settings have options
    df_opts = pd.DataFrame()

    # Get the descriptions if available
    descriptions = (
        settings.descriptions if hasattr(settings, 'descriptions') else {}
    )

    tex = settings.tex if hasattr(settings, 'tex') else {}

    for field_ in fields(settings):
        name = field_.name
        type_ = field_.type

        # Add the options if available
        if name == 'opts':
            # Get options from file
            df_opts = _get_options_from_file(settings.name)
            df_opts = df_opts[['parameter', 'type', 'value', 'description']]
            df_opts.loc[:, 'parameter'] = df_opts['parameter'].apply(
                lambda x: f'{prefix}.opts.{x}',
            )

        # Add the settings to the dictionary
        elif (
            name not in ['info', 'descriptions', 'tex']
            and '_types' not in name
        ):
            # Get the type of the field
            if isinstance(type_, UnionType):
                data['parameter'].append(f'{prefix}.{name}')
                data['type'].append(str(type_))
                data['value'].append(getattr(settings, name))

            elif type_.__name__ == 'list':
                if isinstance(type_.__args__[0], UnionType):
                    type_0 = str(type_.__args__[0])
                else:
                    type_0 = type_.__args__[0].__name__

                data['parameter'].append(f'{prefix}.{name}')
                data['type'].append(f'list[{type_0}]')
                data['value'].append(getattr(settings, name))

            elif type_.__name__ == 'dict':
                values_dict: dict = getattr(settings, name)
                types_dict: dict = getattr(settings, f'_{name}_types')

                for key, val in values_dict.items():
                    data['parameter'].append(f'{prefix}.{name}.{key}')
                    data['type'].append(types_dict[key])
                    data['value'].append(val)

            else:
                data['parameter'].append(f'{prefix}.{name}')
                data['type'].append(type_.__name__)
                data['value'].append(getattr(settings, name))

            # Add the tex if available
            if name in tex:
                data['tex'].append(tex[name])
            else:
                data['tex'].append('')

            # Add the description if available
            if name in descriptions:
                data['description'].append(descriptions[name])
            else:
                data['description'].append('')

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Concatenate the DataFrames
    return pd.concat([df, df_opts], ignore_index=True)


def _get_options_from_file(name: str) -> pd.DataFrame:
    opts_folder = Path(__file__).parent / 'options'
    if name in ['kinsol', 'fast_newton', 'newton']:
        df_1 = pd.read_csv(
            opts_folder / 'casadi_root_finding.csv', na_filter=False
        )

    elif name in ['idas', 'cvodes']:
        df_1 = pd.read_csv(
            opts_folder / 'casadi_integrator.csv', na_filter=False
        )

    elif name in ['ipopt']:
        df_1 = pd.read_csv(opts_folder / 'casadi_nlp.csv', na_filter=False)

    df_2 = pd.read_csv(
        Path(__file__).parent / 'options' / f'{name}.csv', na_filter=False
    )

    return merge_dfs(df_1, df_2)


def merge_dfs(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    # Merge df_1 with df_2 on 'parameter' column
    merged_df = df_1.merge(
        df_2, on='parameter', suffixes=('', '_new'), how='left'
    )

    # Update the 'type', 'value', and 'description' columns
    # in df_1 with those from df_2
    for column in ['type', 'value', 'description']:
        merged_df[column] = merged_df[f'{column}_new'].combine_first(
            merged_df[column]
        )

    # Drop the '_new' columns
    return merged_df.drop(
        columns=[
            f'{column}_new' for column in ['type', 'value', 'description']
        ]
    )


def extract_settings_from_file(
    file_name: str = 'settings_template.csv',
) -> dict:
    # Read the settings file
    df = pd.read_csv(
        Path.cwd() / SETTING_FOLDER / file_name,
        na_filter=False,
    )

    # Convert the DataFrame to a nested dictionary
    # TODO: add type hints
    df_dict = _df_to_nested_dict(df)

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


def _df_to_nested_dict(df: pd.DataFrame) -> dict:
    nested_dict: dict = {}
    tex_dict: dict[str, str] = {}
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

            if 'model.info' in row['parameter']:
                key = row['parameter'].replace('model.info.', '')
                tex_dict[key] = (
                    row['tex'].replace('{', '{{').replace('}', '}}')
                )

    if len(tex_dict) > 0:
        nested_dict['model']['tex'] = tex_dict

    return nested_dict


def _cast_value(value: str, value_type: str):
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
