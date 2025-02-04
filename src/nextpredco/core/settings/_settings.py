from dataclasses import fields
from pathlib import Path
from types import UnionType

import pandas as pd

from nextpredco.core import (
    CONFIG_FOLDER,
    DESCRIPTION,
    PARAMETER,
    ROLE,
    SETTING_FOLDER,
    SS_VARS_PRIMARY,
    SS_VARS_SECONDARY,
    TEX,
    TYPE,
    VALUE,
    logger,
    tools,
)
from nextpredco.core.settings._controller_settings import (
    ControllerSettings,
    MPCSettings,
    PIDSettings,
)
from nextpredco.core.settings._integrator_settings import (
    IDASSettings,
    IntegratorSettings,
    TaylorSettings,
)
from nextpredco.core.settings._model_settings import ModelSettings
from nextpredco.core.settings._observer_settings import (
    KalmanSettings,
    ObserverSettings,
    StateFeedback,
)
from nextpredco.core.settings._optimizer_settings import (
    IPOPTSettings,
    OptimizerSettings,
)
from nextpredco.core.settings._plant_settings import PlantSettings

type Settings = (
    ModelSettings
    | ControllerSettings
    | IntegratorSettings
    | OptimizerSettings
    | PlantSettings
    | ObserverSettings
)


class SettingsFactory:
    @staticmethod
    def create(name, **settings) -> Settings:
        # Model
        if name == 'model':
            return ModelSettings(**settings)

        # Integrators
        if name == 'taylor':
            return TaylorSettings(**settings)

        if name == 'idas':
            return IDASSettings(**settings)

        # Optimizers
        if name == 'ipopt':
            return IPOPTSettings(**settings)

        # Controllers
        if name == 'pid':
            return PIDSettings(**settings)

        if name == 'mpc':
            return MPCSettings(**settings)

        # Observers
        if name == 'state_feedback':
            return StateFeedback(**settings)

        if name == 'kalman':
            return KalmanSettings(**settings)
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
    df_model = _merge_dfs(df1=df_model, df2=df_vars)

    # Create the DataFrames
    settings_dfs: list[pd.DataFrame] = [df_model, df_info]

    # Read the configuration file
    df_configs = _read_config_csv(config_file)
    logger.debug(df_configs)
    input('Press Enter to continue...')
    # Create the INTEGRATOR settings DataFrame
    if _get_value(df=df_configs, parameter='model.is_continuous'):
        settings_dfs.append(
            _get_class_settings(df=df_configs, prefix='model.integrator'),
        )

    # Create the CONTROLLER settings DataFrame
    settings_dfs.append(
        _get_class_settings(df=df_configs, prefix='controller'),
    )

    # Create the INTEGRATOR settings DataFrame
    settings_dfs.append(
        _get_class_settings(df=df_configs, prefix='controller.integrator'),
    )

    # Create the OPTIMIZER settings DataFrame
    settings_dfs.append(
        _get_class_settings(df=df_configs, prefix='controller.optimizer'),
    )

    # Create the CONTROLLER settings DataFrame
    settings_dfs.append(
        _get_class_settings(df=df_configs, prefix='observer'),
    )

    # Create the INTEGRATOR settings DataFrame
    settings_dfs.append(
        _get_class_settings(df=df_configs, prefix='observer.integrator'),
    )

    # Create the OPTIMIZER settings DataFrame
    settings_dfs.append(
        _get_class_settings(df=df_configs, prefix='observer.optimizer'),
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
    settings_df[VALUE] = [
        tools.list_to_str(value) if isinstance(value, list) else value
        for value in settings_df[VALUE]
    ]

    # Lowercase the string values
    settings_df[[PARAMETER, TYPE, VALUE]] = settings_df[
        [PARAMETER, TYPE, VALUE]
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
        PARAMETER: [],
        TYPE: [],
        VALUE: [],
        TEX: [],
        DESCRIPTION: [],
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
        if row[VALUE] != '':
            data[PARAMETER].append('model.info.' + row[PARAMETER])
            data[TYPE].append(row[TYPE])
            data[VALUE].append(_cast_value(row[VALUE], row[TYPE]))
            data[DESCRIPTION].append(row[DESCRIPTION])
            data[TEX].append(row[TEX])

            for ss_var in SS_VARS_PRIMARY:
                if ss_var in row[ROLE] and row[ROLE] != 'const':
                    data_vars[ss_var].append(row[PARAMETER])

            for ss_var in SS_VARS_SECONDARY:
                if ss_var in row[ROLE] and row[ROLE] != 'const':
                    data_vars[ss_var].append(row[PARAMETER])

            if row[ROLE] == 'const':
                data_vars['const'].append(row[PARAMETER])

    data_vars['upq'] = data_vars['u'] + data_vars['p'] + data_vars['q']

    df_vars: dict[str, list] = {
        PARAMETER: [],
        VALUE: [],
    }
    for key, value in data_vars.items():
        df_vars[PARAMETER].append(f'model.{key}_vars')
        df_vars[VALUE].append(value)

    return pd.DataFrame(data), pd.DataFrame(df_vars)


def _merge_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # Merge df1 with df2 on PARAMETER column
    merged_df = df1.merge(df2, on=PARAMETER, suffixes=('', '_new'), how='left')

    # Update the TYPE, VALUE, and DESCRIPTION columns
    # in df1 with those from df2
    for column in df2.columns:
        if column not in [PARAMETER, 'cpp_type']:
            merged_df[column] = merged_df[f'{column}_new'].combine_first(
                merged_df[column]
            )

    # Drop the '_new' columns
    return merged_df.drop(
        columns=[
            f'{column}_new'
            for column in df2.columns
            if column not in [PARAMETER, 'cpp_type']
        ]
    )


def _get_type(df: pd.DataFrame, parameter: str) -> str:
    return df.loc[df[PARAMETER] == parameter, TYPE].values[0]


def _get_value(df: pd.DataFrame, parameter: str) -> str:
    value = df.loc[df[PARAMETER] == parameter, VALUE].values[0]
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
        PARAMETER: [],
        TYPE: [],
        VALUE: [],
        TEX: [],
        DESCRIPTION: [],
    }

    # Create an empty DataFrame for the options
    # in case the settings have options
    df_opts = pd.DataFrame()

    # Get the descriptions if available
    descriptions = (
        settings.descriptions if hasattr(settings, 'descriptions') else {}
    )

    tex = settings.tex if hasattr(settings, TEX) else {}

    for field_ in fields(settings):
        name = field_.name
        type_ = field_.type

        # Add the options if available
        if name == 'opts':
            # Get options from file
            df_opts = _get_options_from_file(name=settings.name, prefix=prefix)

        # Add the settings to the dictionary
        elif (
            name not in ['info', 'descriptions', TEX] and '_types' not in name
        ):
            # Get the type of the field
            if isinstance(type_, UnionType):
                data[PARAMETER].append(f'{prefix}.{name}')
                data[TYPE].append(str(type_))
                data[VALUE].append(getattr(settings, name))

            elif type_.__name__ == 'list':
                if isinstance(type_.__args__[0], UnionType):
                    type_0 = str(type_.__args__[0])
                else:
                    type_0 = type_.__args__[0].__name__

                data[PARAMETER].append(f'{prefix}.{name}')
                data[TYPE].append(f'list[{type_0}]')
                data[VALUE].append(getattr(settings, name))

            elif type_.__name__ == 'dict':
                values_dict: dict = getattr(settings, name)
                types_dict: dict = getattr(settings, f'_{name}_types')

                for key, val in values_dict.items():
                    data[PARAMETER].append(f'{prefix}.{name}.{key}')
                    data[TYPE].append(types_dict[key])
                    data[VALUE].append(val)

            else:
                data[PARAMETER].append(f'{prefix}.{name}')
                data[TYPE].append(type_.__name__)
                data[VALUE].append(getattr(settings, name))

            # Add the tex if available
            if name in tex:
                data[TEX].append(tex[name])
            else:
                data[TEX].append('')

            # Add the description if available
            if name in descriptions:
                data[DESCRIPTION].append(descriptions[name])
            else:
                data[DESCRIPTION].append('')

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Concatenate the DataFrames
    return pd.concat([df, df_opts], ignore_index=True)


def _get_options_from_file(name: str, prefix: str = '') -> pd.DataFrame:
    # CasADi has some common options used in their interfaces
    # to 3rd party solvers. Thus, we need to merge these options
    # with the specific options of the solver.
    # There are 3 groups: root finding, integrator, and NLP solvers.
    opts_folder = Path(__file__).parent / 'options'
    opts_path = Path(__file__).parent / 'options' / f'{name}.csv'

    if not opts_path.exists():
        return pd.DataFrame()

    if name in ['kinsol', 'fast_newton', 'newton']:
        df1 = pd.read_csv(
            opts_folder / 'casadi_root_finding.csv', na_filter=False
        )

    elif name in ['idas', 'cvodes']:
        df1 = pd.read_csv(
            opts_folder / 'casadi_integrator.csv', na_filter=False
        )

    elif name in ['ipopt']:
        df1 = pd.read_csv(opts_folder / 'casadi_nlp.csv', na_filter=False)

    df2 = pd.read_csv(opts_path, na_filter=False)

    df_opts = _merge_dfs(df1, df2)
    df_opts = df_opts[df_opts['frequently_used'] == 'x']
    df_opts = df_opts[[PARAMETER, TYPE, VALUE, DESCRIPTION]]
    df_opts.loc[:, PARAMETER] = df_opts[PARAMETER].apply(
        lambda x: f'{prefix}.opts.{x}',
    )
    return df_opts


def read_settings_csv(
    file_name: str = 'settings_template.csv',
) -> dict:
    # Read the settings file
    df = pd.read_csv(
        Path.cwd() / SETTING_FOLDER / file_name,
        na_filter=False,
    )

    # Convert the DataFrame to a nested dictionary
    # TODO: add type hints
    d = _df_to_nested_dict(df)

    # Put nested settings into groups
    all_: dict = {}
    all_ |= _get_settings(dict_=d, parent='model', children=['integrator'])
    all_ |= _get_settings(
        dict_=d, parent='controller', children=['optimizer', 'integrator']
    )
    all_ |= _get_settings(
        dict_=d, parent='observer', children=['optimizer', 'integrator']
    )

    # Get PLANT settings
    if 'plant' in d:
        pass
    else:
        all_['plant'] = None

    return all_


def _df_to_nested_dict(df: pd.DataFrame) -> dict:
    nested_dict: dict = {}
    tex_dict: dict[str, str] = {}
    child_dict: dict = {}
    for _, row in df.iterrows():
        if row[VALUE] != '':
            keys = row[PARAMETER].split('.', 3)
            value_type = row[TYPE]
            value = _cast_value(row[VALUE], value_type)

            d = nested_dict
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d[keys[-1]] = value

            # Since TEX is a column in the DataFrame, we need to
            # convert it to a dictionary and add it to the nested dict
            # TODO: Add latex for other settings
            if 'model.info' in row[PARAMETER]:
                key = row[PARAMETER].replace('model.info.', '')
                tex_dict[key] = row[TEX].replace('{', '{{').replace('}', '}}')

    if len(tex_dict) > 0:
        nested_dict['model'][TEX] = tex_dict

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


def _get_settings(dict_: dict, parent: str, children: list[str]) -> dict:
    d: dict = {}
    if parent in dict_:
        for child in children:
            if child in dict_[parent]:
                d[parent + '.' + child] = SettingsFactory.create(
                    **dict_[parent][child]
                )
                dict_[parent].pop(child)
            else:
                d[parent + '.' + child] = None

        # Get MODEL settings
        d[parent] = SettingsFactory().create(**dict_[parent])
    else:
        d[parent] = None
    return d
