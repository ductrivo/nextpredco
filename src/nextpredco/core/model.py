import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nextpredco.core.consts import (
    DATA_DIR,
    SS_VARS_DB,
    SS_VARS_PRIMARY,
    SS_VARS_SECONDARY,
    SS_VARS_SOURCES,
)
from nextpredco.core.custom_types import SourceType
from nextpredco.core.descriptors import ReadOnly, SystemVariable
from nextpredco.core.errors import SystemVariableError
from nextpredco.core.logger import logger


class Model:
    k = ReadOnly()
    x = SystemVariable()
    z = SystemVariable()
    u = SystemVariable()
    p = SystemVariable()
    q = SystemVariable()
    y = SystemVariable()
    m = SystemVariable()
    o = SystemVariable()
    upq = SystemVariable()

    def __init__(
        self,
        model_data_path: Path = DATA_DIR / 'model_cstr.csv',
        t_max: float = 10,
        t_clock: float = 1,
        t_samp: float = 0.1,
        sources: SourceType = (
            'goal',
            'est',
            'act',
            'meas',
            'filt',
            'meas_clock',
            'filt_clock',
        ),
    ) -> None:
        self._k: int = 0
        self._k_clock: int = 0

        # Extract model information and initial values
        self._model_info = pd.read_csv(model_data_path)
        self._sys_vars: list[str] = list(self._model_info['variable'])
        logger.debug(
            'Model information extracted, system variables:\n%s',
            self._sys_vars,
        )

        # Extract state space variables
        for ss_var in SS_VARS_PRIMARY + SS_VARS_SECONDARY:
            value = self._extract_vars(ss_var)
            setattr(self, f'_{ss_var}_vars', value)
        logger.debug('State space variables extracted.')

        self._upq_vars: list[str] = self._u_vars + self._p_vars + self._q_vars  # type: ignore[attr-defined]
        self._primary_vars = self._get_primary_vars()
        self._validate_definitions()
        logger.debug('Model definitions validated.')

        # Extract time information
        self._n_max = round(t_max / t_samp)
        self._n_clock_max = round(t_max / t_clock)

        # Extract sources information
        self._sources = sources

        # Create full arrays for each source
        created_attrs: list[str] = []
        for ss_var in SS_VARS_DB:
            vars_list = getattr(self, f'_{ss_var}_vars')
            # TODO: May have different initial values for each source
            init_val = self._extract_init_val(vars_list)
            vec_length = len(vars_list)

            # Create full arrays for each source
            for source in sources:
                n_max = self._n_clock_max if 'clock' in source else self._n_max
                arr_full = np.zeros((vec_length, n_max))
                arr_full[:, 0, None] += init_val
                attr_name = f'_{ss_var}_{source}_full'
                setattr(self, attr_name, arr_full)
                created_attrs.append(attr_name)

        logger.debug('Created attributes:\n%s', sorted(created_attrs))

    def _validate_definitions(self):
        self._check_length(
            list1=self._primary_vars,
            list2=self._sys_vars,
            name1='_primary_vars',
            name2='_sys_vars',
        )
        self._check_subset(
            ss_vars_subset=SS_VARS_PRIMARY,
            ss_vars_superset=self._sys_vars,
            superset_name='system',
        )
        self._check_subset(
            ss_vars_subset=SS_VARS_SECONDARY,
            ss_vars_superset=self._primary_vars,
            superset_name='primary',
        )

    def _check_length(
        self,
        list1: list[str],
        list2: list[str],
        name1: str,
        name2: str,
    ):
        if len(list1) != len(list2):
            raise ValueError(  # noqa: TRY003
                f'The length of {name1} must be equal to'
                f'the length of {name2}.',
            )

    def _check_subset(
        self,
        ss_vars_subset: list[str],
        ss_vars_superset: list[str],
        superset_name: str,
    ):
        # ss_var: state space variable
        for ss_var in ss_vars_subset:
            var_vars = getattr(self, f'_{ss_var}_vars')
            if not set(var_vars).issubset(set(ss_vars_superset)):
                raise SystemVariableError(ss_var, superset_name)

    def _get_primary_vars(self) -> list[str]:
        primary_vars = []
        for var_ in SS_VARS_PRIMARY:
            primary_vars.extend(getattr(self, f'_{var_}_vars'))

        return primary_vars

    def _extract_vars(self, name: str) -> list[str]:
        return self._model_info[
            self._model_info['state_space'].str.contains(name)
        ]['variable'].tolist()

    def _extract_init_val(self, vars_list: list[str]) -> NDArray[np.float64]:
        value_vec = []
        for var in vars_list:
            value = self._model_info.loc[
                self._model_info['variable'] == var,
                'value',
            ].values[0]

            value_vec.append(value)

        return np.array([value_vec]).T

    def n(self, ss_var: str) -> int:
        return len(getattr(self, f'_{ss_var}_vars'))


if __name__ == '__main__':
    model = Model()
    print(model.x.act.val)
