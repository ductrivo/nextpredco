import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nextpredco.core._consts import SS_VARS_DB, SS_VARS_SECONDARY
from nextpredco.core._logger import logger
from nextpredco.core._typing import SourceType
from nextpredco.core.errors import (
    EmptyArrayError,
    InvalidK0ValueError,
    NotAvailableAttributeError,
    ReadOnlyAttributeError,
)


class ReadOnly:
    def __set_name__(self, owner, name: str) -> None:
        self._name = name

    def __set__(self, instance, value) -> None:
        raise ReadOnlyAttributeError(self._name)

    def __get__(self, instance, owner):
        return getattr(instance._settings, f'{self._name}')


class ReadOnlyData:
    def __set_name__(self, owner, name: str) -> None:
        self._name = name

    def __set__(self, instance, value) -> None:
        raise ReadOnlyAttributeError(self._name)

    def __get__(self, instance, owner):
        return getattr(instance, f'_{self._name}')


class ReadOnlyInt(ReadOnly):
    def __get__(self, instance, owner) -> int:
        return getattr(instance._settings, f'{self._name}')


class ReadOnlyFloat(ReadOnly):
    def __get__(self, instance, owner) -> float:
        return getattr(instance._settings, f'{self._name}')


class ReadOnlyStr(ReadOnly):
    def __get__(self, instance, owner) -> str:
        return getattr(instance._settings, f'{self._name}')


class ReadOnlySource(ReadOnly):
    def __get__(self, instance, owner) -> SourceType:
        return getattr(instance._settings, f'{self._name}')


class ReadOnlyPandas(ReadOnly):
    def __get__(self, instance, owner) -> pd.DataFrame:
        return getattr(instance._settings, f'{self._name}')


class SystemVariableView:
    def __init__(
        self,
        k: int,
        arr_full: NDArray,
        idx_list: list[int],
    ) -> None:
        super().__init__()
        self._k = k
        self._arr_full = arr_full
        self._idx_list = idx_list

    @property
    def val(self) -> NDArray:
        if len(self._idx_list) == 0:
            return np.array([[]]).T

        if len(self._idx_list) == self._arr_full.shape[0]:
            return self._arr_full[:, self._k, None]

        return self._arr_full[self._idx_list, self._k, None]

    @val.setter
    def val(self, val: NDArray) -> None:
        if len(self._idx_list) == 0:
            if val.size == 0:
                return
            logger.warning('Trying to set a value to an empty array.')
        self._arr_full[self._idx_list, self._k, None] = val

    @property
    def last(self) -> NDArray:
        if len(self._idx_list) == 0:
            return np.array([[]]).T

        k = self._k - 1 if self._k > 0 else 0

        if len(self._idx_list) == self._arr_full.shape[0]:
            return self._arr_full[:, k, None]

        return self._arr_full[self._idx_list, k, None]

    @property
    def hist(self) -> NDArray:
        if len(self._idx_list) == 0:
            return np.array([[]]).T

        if len(self._idx_list) == self._arr_full.shape[0]:
            return self._arr_full[:, : (self._k + 1)]

        return self._arr_full[self._idx_list, : (self._k + 1)]

    @property
    def full(self) -> NDArray:
        if len(self._idx_list) == 0:
            return np.array([[]]).T
        return self._arr_full

    def get_val(self, k: int) -> NDArray:
        if len(self._idx_list) == 0:
            return np.array([[]]).T
        return self._arr_full[self._idx_list, k, None]

    def set_val(self, k: int, val: NDArray | float) -> None:
        if len(self._idx_list) == 0:
            raise EmptyArrayError()

        if len(self._idx_list) == self._arr_full.shape[0]:
            self._arr_full[:, k, None] = val

        self._arr_full[self._idx_list, k, None] = val

    def get_hist(
        self,
        k0: int,
        k1: int,
    ) -> NDArray:
        if k0 < 0:
            raise InvalidK0ValueError

        if len(self._idx_list) == 0:
            return np.array([[]]).T

        if len(self._idx_list) == self._arr_full.shape[0]:
            return self._arr_full[:, k0 : (k1 + 1)]

        return self._arr_full[self._idx_list, k0 : (k1 + 1)]

    def set_hist(
        self,
        k0: int,
        k1: int,
        val: NDArray,
    ) -> None:
        if k0 < 0:
            raise InvalidK0ValueError

        if len(self._idx_list) == 0:
            raise EmptyArrayError()

        if len(self._idx_list) == self._arr_full.shape[0]:
            self._arr_full[:, k0 : (k1 + 1)] = val

        self._arr_full[self._idx_list, k0 : (k1 + 1)] = val


class SystemVariableViewPreds:
    def __init__(
        self,
        k: int,
        preds_full: dict[int, NDArray],
        idx_list: list[int],
    ) -> None:
        self._k = k
        self._preds_full = preds_full
        self._idx_list = idx_list

    @property
    def horizon(self) -> NDArray:
        return self.get_horizon(self._k)

    @horizon.setter
    def horizon(self, val: NDArray):
        self.set_horizon(self._k, val)

    @property
    def val(self) -> NDArray:
        return self.get_val(self._k)

    @val.setter
    def val(self, val: NDArray):
        self.set_val(self._k, val)

    def get_horizon(self, k: int) -> NDArray:
        arr = self._preds_full[k]

        # TODO: is this necessary?
        if len(self._idx_list) == 0:
            return np.array([[]])

        if len(self._idx_list) == arr.shape[0]:
            return arr

        return arr[self._idx_list, :]

    def set_horizon(
        self,
        k: int,
        val: NDArray,
        k_preds: list[int] | int | None = None,
    ) -> None:
        if len(self._idx_list) == 0:
            raise EmptyArrayError()

        if (k_preds is None) or (k not in self._preds_full):
            self._preds_full[k] = val
        else:
            k_preds_ = [k_preds] if isinstance(k_preds, int) else k_preds
            self._preds_full[k][self._idx_list, k_preds_] = val

    def get_val(self, k: int) -> NDArray:
        return self.get_horizon(k)[:, 0, None]

    def set_val(
        self,
        k: int,
        val: NDArray,
        k_pred: list[int] | int | None = None,
    ):
        k_pred_ = [self._k] if k_pred is None else k_pred
        if k in self._preds_full:
            self.set_horizon(k, val, k_preds=k_pred_)
        # TODO: standardize the val type


class VariableSource(ReadOnlyData):
    goal: SystemVariableView | ReadOnlyData = ReadOnlyData()
    act: SystemVariableView | ReadOnlyData = ReadOnlyData()
    est: SystemVariableView | ReadOnlyData = ReadOnlyData()
    meas: SystemVariableView | ReadOnlyData = ReadOnlyData()
    filt: SystemVariableView | ReadOnlyData = ReadOnlyData()
    preds: SystemVariableViewPreds | ReadOnlyData = ReadOnlyData()

    goal_clock: SystemVariableView | ReadOnlyData = ReadOnlyData()
    act_clock: SystemVariableView | ReadOnlyData = ReadOnlyData()
    est_clock: SystemVariableView | ReadOnlyData = ReadOnlyData()
    meas_clock: SystemVariableView | ReadOnlyData = ReadOnlyData()
    filt_clock: SystemVariableView | ReadOnlyData = ReadOnlyData()

    def __init__(
        self,
        k: int,
        k_clock: int,
        arr_fulls: dict[str, NDArray | dict[int, NDArray]],
        idx_lists: dict[str, list[int]],
    ) -> None:
        for source, arr_full in arr_fulls.items():
            if source == 'preds' and isinstance(arr_full, dict):
                idx_list = idx_lists[source]
                attr_name = f'_{source}'
                self._preds = SystemVariableViewPreds(
                    k, arr_full, idx_lists[source]
                )
            elif isinstance(arr_full, np.ndarray):
                k_ = k_clock if 'clock' in source else k
                idx_list = idx_lists[source]
                attr_name = f'_{source}'
                setattr(
                    self,
                    attr_name,
                    SystemVariableView(k_, arr_full, idx_list),
                )


class SystemVariable:
    def __set_name__(self, owner, name: str) -> None:
        self._name = name
        self._private_name = f'_{name}'

    def __set__(self, instance, value) -> None:
        error_msg = (
            'Please specify the source i.e. "est", "act", "meas", "filt", etc.'
        )
        raise AttributeError(error_msg)

    def __get__(self, instance, owner) -> VariableSource:
        sources: SourceType = instance._settings.sources

        self._vars: list[str] = getattr(
            instance._settings,
            f'{self._name}_vars',
        )

        arr_fulls: dict[str, NDArray | dict[int, NDArray]] = {}
        idx_lists: dict[str, list[int]] = {}
        for source in sources:
            if source == 'preds' and self._name == 'x':
                arr_fulls[source] = instance._data.x_preds_full
                idx_lists[source] = list(range(instance.n('x')))
            elif source == 'preds' and self._name == 'u':
                arr_fulls[source] = instance._data.u_preds_full
                idx_lists[source] = list(range(instance.n('u')))
            elif source == 'preds':
                continue
            else:
                arr_fulls[source], idx_lists[source] = (
                    self._get_arr_fulls_and_idx_lists(
                        instance,
                        ss_name=self._name,
                        name=self._name,
                        source=source,
                    )
                )

        return VariableSource(
            instance._data.k,
            instance._data.k_clock,
            arr_fulls,
            idx_lists,
        )

    def _get_arr_fulls_and_idx_lists(
        self,
        instance,
        ss_name: str,
        name: str,
        source: str,
    ) -> tuple[NDArray, list[int]]:
        # Extract the variables
        x_vars: list[str] = instance._settings.x_vars
        z_vars: list[str] = instance._settings.z_vars
        upq_vars: list[str] = instance._settings.upq_vars

        # Extract the full arrays
        if name in ['x', *x_vars, *SS_VARS_SECONDARY]:
            arr_full = getattr(instance._data, f'x_{source}_full')
        elif name in ['z', *z_vars]:
            arr_full = getattr(instance._data, f'z_{source}_full')
        elif name in ['u', 'p', 'q', 'upq', *upq_vars]:
            arr_full = getattr(instance._data, f'upq_{source}_full')
        else:
            raise NotAvailableAttributeError(name)

        # Extract the indexes
        if name in SS_VARS_DB:
            idx_list = list(range(arr_full.shape[0]))

        elif name in ['u', 'p', 'q']:
            vars_ = getattr(instance._settings, f'{name}_vars')
            idx_list = [upq_vars.index(var_) for var_ in vars_]

        elif name in SS_VARS_SECONDARY:
            vars_ = getattr(instance._settings, f'{name}_vars')
            idx_list = [x_vars.index(var_) for var_ in vars_]

        elif name in x_vars:
            idx_list = [x_vars.index(name)]

        elif name in z_vars:
            idx_list = [z_vars.index(name)]

        elif name in upq_vars:
            idx_list = [upq_vars.index(name)]

        else:
            raise NotAvailableAttributeError(name)

        return arr_full, idx_list


class PhysicalVariable(SystemVariable):
    def __get__(self, instance, owner):
        self._x_vars: list[str] = instance._settings.x_vars  # type: ignore[annotation-unchecked]
        self._z_vars: list[str] = instance._settings.z_vars  # type: ignore[annotation-unchecked]
        self._upq_vars: list[str] = instance._settings.upq_vars  # type: ignore[annotation-unchecked]

        self._sources: SourceType = instance._settings.sources  # type: ignore[annotation-unchecked]
        self._instance = instance
        return self.get_val

    def get_val(self, var_: str):
        if var_ in self._x_vars:
            ss_name = 'x'
        elif var_ in self._z_vars:
            ss_name = 'z'
        elif var_ in self._upq_vars:
            ss_name = 'upq'

        arr_fulls: dict[str, NDArray | dict[int, NDArray]] = {}
        idx_lists: dict[str, list[int]] = {}
        for source in self._sources:
            if source == 'preds':
                continue
            arr_fulls[source], idx_lists[source] = (
                self._get_arr_fulls_and_idx_lists(
                    self._instance,
                    ss_name=ss_name,
                    name=var_,
                    source=source,
                )
            )
        return VariableSource(
            self._instance._data.k,
            self._instance._data.k_clock,
            arr_fulls,
            idx_lists,
        )


class TimeVariable:
    def __set_name__(self, owner, name: str) -> None:
        self._name = name
        self._private_name = f'_{name}'

    def __set__(self, instance, value) -> None:
        error_msg = 'Please model.t.set_val or model.t.set_hist.'
        raise AttributeError(error_msg)

    def __get__(self, instance, owner) -> SystemVariableView:
        if self._name == 't':
            arr_full = instance._data.t_full
        elif self._name == 't_clock':
            arr_full = instance._data.t_clock_full

        return SystemVariableView(
            k=instance._data.k,
            arr_full=arr_full,
            idx_list=[0],
        )
