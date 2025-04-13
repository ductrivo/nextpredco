from collections import OrderedDict as OrderedDict
from collections.abc import Callable
from functools import reduce as reduce
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rich.pretty import pretty_repr as pretty_repr

from nextpredco.core._consts import (
    COST_ELEMENTS,
    PREDICTION_ELEMENTS,
    SS_VARS_DB,
    SS_VARS_PRIMARY,
    SS_VARS_SECONDARY,
)

# from nextpredco.core._data2 import PredictionsData
from nextpredco.core._logger import logger as logger
from nextpredco.core._typing import (
    Array2D,
    FloatType,
    IntType,
    PredDType,
    SourceType,
    isArray2D,
    isIntType,
)
from nextpredco.core.errors import (
    EmptyArrayError,
    InvalidK0ValueError,
    NotAvailableAttributeError,
    ReadOnlyAttributeError,
)

if TYPE_CHECKING:
    from nextpredco.core._data import ModelData
    from nextpredco.core.controller._mpc import (
        MPC,
        BoundsData,
        CostsData,
        GoalsData,
        MPCData,
        PredictionsData,
    )
    from nextpredco.core.integrator import IDAS, Integrator, Taylor
    from nextpredco.core.model._model import ModelABC
    from nextpredco.core.settings import ModelSettings, MPCSettings
from nextpredco.core._sync import GlobalState

ReadOnlyType = TypeVar(
    'ReadOnlyType',
    SourceType,
    list[str],
    IntType,
    FloatType,
    'VariableView',
    'VariableViewDict',
    'CostStructure',
    'ModelSettings',
    'ModelData',
    'Integrator',
    'IDAS',
    'Taylor',
    'ModelABC',
    'MPCSettings',
    'MPCData',
    'CostsData',
    'PredictionsData',
    'GoalsData',
    'BoundsData',
)


def _get_full_data_and_indexes(
    instance: 'ModelABC',
    ss_name: str,
    name: str,
) -> tuple[Array2D, list[IntType], list[str]]:
    # Define the variable names
    x_vars = instance.settings.x_vars
    z_vars = instance.settings.z_vars
    upq_vars = instance.settings.upq_vars
    full_data: Array2D

    # Get the full data
    if name in ['x', *x_vars, *SS_VARS_SECONDARY]:
        full_data = instance.data.x_full
    elif name in ['z', *z_vars]:
        full_data = instance.data.z_full
    elif name in ['u', 'p', 'q', 'upq', *upq_vars]:
        full_data = instance.data.upq_full
    else:
        raise NotAvailableAttributeError(name)

    # Get the index list
    if name in SS_VARS_DB:
        indexes = list(range(full_data.shape[0]))

    elif name in ['u', 'p', 'q']:
        vars_ = getattr(instance.settings, f'{name}_vars')
        indexes = [upq_vars.index(var_) for var_ in vars_]

    elif name in SS_VARS_SECONDARY:
        vars_ = getattr(instance.settings, f'{name}_vars')
        indexes = [x_vars.index(var_) for var_ in vars_]

    # This is for physical variables
    elif name in x_vars:
        indexes = [x_vars.index(name)]

    elif name in z_vars:
        indexes = [z_vars.index(name)]

    elif name in upq_vars:
        indexes = [upq_vars.index(name)]

    else:
        raise NotAvailableAttributeError(name)

    # Get column names
    if name in [*SS_VARS_PRIMARY, 'upq']:
        columns = getattr(instance.settings, f'{name}_vars')
    else:
        columns = [name]
    return full_data, indexes, columns


class ReadOnly(Generic[ReadOnlyType]):
    """A descriptor for read-only attributes."""

    def __init__(self, data_container_name: str = '') -> None:
        """Initialize the descriptor.

        Parameters
        ----------
        data_container_name : str
            The name of the data container that stores the data
            for this attribute. If not provided, the attribute
            will be stored in the instance's __dict__.
        """
        self._data_container_name = data_container_name

    def __set_name__(self, owner: type, name: str) -> None:
        """Set the name of the descriptor."""
        self._name = name

    def __set__(self, instance: object, value: ReadOnlyType) -> None:
        """Raise a ReadOnlyAttributeError when trying to
        set a read-only attribute.
        """
        raise ReadOnlyAttributeError(self._name)

    def __get__(self, instance, owner: type) -> ReadOnlyType:
        """
        Get the value of the read-only attribute.

        Parameters
        ----------
        instance : object
            The instance of the class that owns this descriptor.
        owner : type
            The class that owns this descriptor.

        Returns
        -------
        ReadOnlyType
            The value of the read-only attribute.

        Raises
        ------
        AttributeError
            If the data container is invalid.
        """
        data_container_name = self._data_container_name
        if data_container_name == '':
            # If the data container name is empty, it means that
            # the data is stored in the instance's __dict__.
            return getattr(instance, f'_{self._name}')
        if data_container_name == 'settings':
            # If the data container name is 'settings', it means that
            # the data is stored in the instance's settings.
            return getattr(instance._settings, self._name)
        if data_container_name == 'data':
            # If the data container name is 'data', it means that
            # the data is stored in the instance's data.
            return getattr(instance.data, self._name)

        if data_container_name == 'model_data':
            # If the data container name is 'data', it means that
            # the data is stored in the instance's data.
            return getattr(instance.model.data, self._name)

        name_list = data_container_name.split('.')
        attr = instance
        for name in name_list:
            attr = getattr(attr, name)

        return getattr(attr, self._name)
        # msg = f'Invalid data container: {data_container_name}'
        # raise AttributeError(msg)


class VariableView:
    def __init__(
        self,
        k: IntType,
        full_data: Array2D,
        indexes: list[IntType],
        columns: list[str],
    ) -> None:
        self._k = k
        self._full_data = full_data
        self._indexes = indexes
        self._columns = columns

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.hist.T, columns=self._columns)

    @property
    def vec(self) -> Array2D:
        return self.get_val(self._k)

    @vec.setter
    def vec(self, val: Array2D) -> None:
        self.set_val(self._k, val)

    @property
    def last(self) -> Array2D:
        return self.get_val(self._k - 1)

    @last.setter
    def last(self, val: Array2D) -> None:
        self.set_val(self._k - 1, val)

    @property
    def next(self) -> Array2D:
        return self.get_val(self._k + 1)

    @next.setter
    def next(self, val: Array2D) -> None:
        self.set_val(self._k + 1, val)

    @property
    def hist(self) -> Array2D:
        return self.get_hist(0, self._k)

    @hist.setter
    def hist(self, val: Array2D) -> None:
        self.set_hist(0, self._k, val)

    @property
    def full(self) -> Array2D:
        return self._full_data

    @full.setter
    def full(self, val: Array2D) -> None:
        self._full_data = val

    def get_val(self, k: IntType) -> Array2D:
        return self.get_hist(k0=k, k1=k)

    def set_val(self, k: IntType, val: Array2D) -> None:
        self.set_hist(k0=k, k1=k, val=val)

    def get_hist(
        self,
        k0: IntType,
        k1: IntType,
    ) -> Array2D:
        if k0 < 0:
            msg = f'k0 must be greater than or equal to 0. You entered {k0}.'
            raise ValueError(msg)

        if len(self._indexes) == 0:
            return np.array([[]]).T

        if len(self._indexes) == self._full_data.shape[0]:
            return self._full_data[:, k0 : (k1 + 1)]

        return self._full_data[self._indexes, k0 : (k1 + 1)]

    def set_hist(
        self,
        k0: IntType,
        k1: IntType,
        val: Array2D,
    ) -> None:
        if not isArray2D(val):
            msg = f'The value must be a 2D array. You entered {type(val)}.'
            raise TypeError(msg)

        if k0 < 0:
            msg = f'k0 must be greater than or equal to 0. You entered {k0}.'
            raise ValueError(msg)

        if len(self._indexes) == self._full_data.shape[0]:
            self._full_data[:, k0 : (k1 + 1)] = val
        else:
            self._full_data[self._indexes, k0 : (k1 + 1)] = val


class VariableViewDict:
    def __init__(
        self,
        k: IntType,
        full_data: PredDType,
        indexes: list[IntType],
        columns: list[str],
    ) -> None:
        self._k = k
        self._full_data = full_data
        self._indexes = indexes
        self._columns = columns

    @property
    def arr(self) -> Array2D:
        try:
            return self.get_val(self._k)
        except ValueError:
            return self.get_val(-1)

    @arr.setter
    def arr(self, val: Array2D) -> None:
        """
        Set the value of the variable view dictionary at the current time step.

        Parameters
        ----------
        val : Array2D
            The value to set.
        """
        self.set_val(self._k, val)

    @property
    def val(self) -> Array2D:
        """
        Get the current value of the variable view dictionary.

        Returns
        -------
        Array2D
            The current value of the variable view dictionary.
        """
        return self.arr[:, 0, None]

    def get_val(self, k: IntType) -> Array2D:
        """
        Retrieve the value of the variable view dictionary at a specified time step.

        Parameters
        ----------
        k : IntType
            The time step for which the value is to be retrieved.

        Returns
        -------
        Array2D
            The value of the variable view dictionary at the specified time step.
        """

        ordered = OrderedDict(self._full_data)
        if k not in self._full_data and k != -1:
            msg = f'The time step {k} is not in the data.'
            raise ValueError(msg)

        if k == -1 and self._indexes is None:
            return next(reversed(ordered.values()))

        if k == -1 and self._indexes is not None:
            return next(reversed(ordered.values()))[self._indexes, :]

        if self._indexes is None:
            return self._full_data[k]

        if len(self._indexes) == 0:
            return np.array([[]])

        return self._full_data[k][self._indexes, :]

    def set_val(self, k: IntType, val: Array2D) -> None:
        """
        Set the value of the variable view dictionary at a specified time step.

        Parameters
        ----------
        k : IntType
            The time step for which the value is to be set.
        val : Array2D
            The value to set.
        """
        # Check if the value is a 2D array
        if not isArray2D(val):
            msg = f'The value must be a 2D array. You entered {type(val)}.'
            raise TypeError(msg)

        # Set the value
        self._full_data[k] = val

    @property
    def df(self) -> pd.DataFrame:
        """
        Stack the values of the full_data dictionary vertically.

        Returns
        -------
        NDArray
            The stacked values.
        """
        time_steps = np.vstack(list(self._full_data.keys()))
        df_arr = np.stack(list(self._full_data.values()))
        n_pred = df_arr.shape[2]

        df_arr = df_arr.reshape((time_steps.shape[0], -1))
        data = np.hstack([time_steps, df_arr])

        columns = (
            self._columns
            if n_pred == 1
            else [
                f'{col}_{step}'
                for col in self._columns
                for step in np.arange(n_pred)
            ]
        )

        df = pd.DataFrame(
            data=data,
            columns=['k', *columns],
        )
        df['k'] = df['k'].astype(int)

        return df


class VariableViewDict2:
    def __init__(
        self,
        k: IntType,
        full_data: PredDType,
        indexes: list[IntType],
        columns: list[str],
    ) -> None:
        self._k = k
        self._full_data = full_data
        self._indexes = indexes
        self._columns = columns

        # logger.debug(
        #     f'\nfull_data = {self._full_data}\nindexes = {self._indexes}\ncolumns = {self._columns}'
        # )

    @property
    def arr(self) -> Array2D:
        return self.get_arr(-1)

    @arr.setter
    def arr(self, arr: Array2D) -> None:
        self.set_arr(self._k, arr)

    @property
    def vec(self) -> Array2D:
        return self.arr[:, 0, None]

    def get_arr(self, k: IntType) -> Array2D:
        ordered = OrderedDict(self._full_data)
        if k not in self._full_data and k != -1:
            msg = f'The time step {k} is not in the data.'
            raise ValueError(msg)

        if k == -1 and self._indexes is None:
            return next(reversed(ordered.values()))

        if k == -1 and self._indexes is not None:
            return next(reversed(ordered.values()))[self._indexes, :]

        if self._indexes is None:
            return self._full_data[k]

        if len(self._indexes) == 0:
            return np.array([[]])

        return self._full_data[k][self._indexes, :]

    def set_arr(self, k: IntType, arr: Array2D) -> None:
        # Check if the value is a 2D array
        if not isArray2D(arr):
            msg = f'The value must be a 2D array. You entered {type(arr)}.'
            raise TypeError(msg)

        # Set the value
        self._full_data[k][self._indexes] = arr

    @property
    def df(self) -> pd.DataFrame:
        time_steps = np.vstack(list(self._full_data.keys()))
        df_arr = np.stack(list(self._full_data.values()))[:, self._indexes, :]
        n_pred = df_arr.shape[2]

        df_arr = df_arr.reshape((time_steps.shape[0], -1))
        data = np.hstack([time_steps, df_arr])

        columns = (
            self._columns
            if n_pred == 1
            else [
                f'{col}_{step}'
                for col in self._columns
                for step in np.arange(n_pred)
            ]
        )

        df = pd.DataFrame(
            data=data,
            columns=['k', *columns],
        )
        df['k'] = df['k'].astype(int)
        return df


# class StateSpaceStructure:
#     """
#     A class to hold the state space structure for a model.

#     It contains the full data arrays for each source, as well as the index lists
#     for each source. It also contains the current time step and the time step
#     in clock time.

#     Attributes
#     ----------
#     goal : VariableView
#         The state space structure for the goal data.
#     act : VariableView
#         The state space structure for the actual data.
#     est : VariableView
#         The state space structure for the estimated data.
#     meas : VariableView
#         The state space structure for the measurement data.
#     filt : VariableView
#         The state space structure for the filtered data.
#     preds : dict[str, VariableView]
#         A dictionary with the state space structures for the predicted data
#         for each source.
#     goal_clock : VariableView
#         The state space structure for the goal data in clock time.
#     act_clock : VariableView
#         The state space structure for the actual data in clock time.
#     est_clock : VariableView
#         The state space structure for the estimated data in clock time.
#     meas_clock : VariableView
#         The state space structure for the measurement data in clock time.
#     filt_clock : VariableView
#         The state space structure for the filtered data in clock time.
#     """

#     goal = ReadOnly[VariableView]()
#     act = ReadOnly[VariableView]()
#     est = ReadOnly[VariableView]()
#     meas = ReadOnly[VariableView]()
#     filt = ReadOnly[VariableView]()
#     preds = ReadOnly[VariableViewDict]()

#     goal_clock = ReadOnly[VariableView]()
#     act_clock = ReadOnly[VariableView]()
#     est_clock = ReadOnly[VariableView]()
#     meas_clock = ReadOnly[VariableView]()
#     filt_clock = ReadOnly[VariableView]()

#     def __init__(
#         self,
#         instance: 'ModelABC',
#         name: str,
#         full_data_dict: dict[str, Array2D],
#         indexes_dict: dict[str, list[IntType]],
#     ) -> None:
#         # Iterate over the sources and create the VariableView instances
#         for source, full_data in full_data_dict.items():
#             # Determine the time step
#             k = instance.k_clock if 'clock' in source else instance.k

#             if name == 'x':
#                 columns = [f'{name_}_{source}' for name_ in instance.x_vars]
#             elif name == 'z':
#                 columns = [f'{name_}_{source}' for name_ in instance.z_vars]
#             elif name == 'upq':
#                 columns = [f'{name_}_{source}' for name_ in instance.upq_vars]
#             elif name == 'u':
#                 columns = [f'{name_}_{source}' for name_ in instance.u_vars]
#             elif name == 'p':
#                 columns = [f'{name_}_{source}' for name_ in instance.p_vars]
#             elif name == 'q':
#                 columns = [f'{name_}_{source}' for name_ in instance.q_vars]
#             # elif name == 'y':
#             #     columns = [f'{name_}_{source}' for name_ in instance.y_vars]
#             else:
#                 columns = [f'{name}_{source}']
#             # Create the VariableView instance
#             indexes = indexes_dict[source]
#             variable_view = VariableView(
#                 k=k,
#                 full_data=full_data,
#                 indexes=indexes,
#                 columns=columns,
#             )

#             # Set the attribute
#             setattr(self, f'_{source}', variable_view)


class CostStructure:
    def __str__(self) -> str:
        """
        Return a string representation of the cost structure.

        Returns
        -------
        str
            A string representation of the cost structure.
        """
        lines = [
            'Cost elements are:',
            f'\tx: {self.x.val}',
            f'\ty: {self.y.val}',
            f'\tu: {self.u.val}',
            f'\tdu: {self.du.val}',
            f'\ttotal: {self.total.val}',
        ]
        return '\n'.join(lines)

    x = ReadOnly[VariableViewDict]()
    y = ReadOnly[VariableViewDict]()
    u = ReadOnly[VariableViewDict]()
    du = ReadOnly[VariableViewDict]()
    total = ReadOnly[VariableViewDict]()

    def __init__(self, k: IntType, full_data: dict[str, PredDType]) -> None:
        """
        Initialize the cost structure.

        Parameters
        ----------
        k : IntType
            The current time step.
        full_data : dict[str, PredDType]
            A dictionary containing the full data arrays for each cost element.
        """
        self._full_data = full_data
        for cost_element in COST_ELEMENTS:
            variable_view_dict = VariableViewDict(
                k=k,
                full_data=full_data[cost_element],
                indexes=[0],
                columns=[f'cost_{cost_element}'],
            )
            setattr(self, f'_{cost_element}', variable_view_dict)

    @property
    def val(self) -> dict[str, Array2D]:
        """
        Return a dictionary containing the current values of the cost elements.

        Returns
        -------
        dict[str, Array2D]
            A dictionary containing the current values of the cost elements.
            The keys are 'x', 'y', 'u', 'du', and 'total', which map to the
            respective cost elements stored within the structure.
        """
        return {
            'x': self.x.val,
            'y': self.y.val,
            'u': self.u.val,
            'du': self.du.val,
            'total': self.total.val,
        }

    @val.setter
    def val(self, new_values: dict[str, Array2D]) -> None:
        """
        Set the current values of the cost elements.

        Parameters
        ----------
        new_values : dict[str, Array2D]
            A dictionary containing the new values of the cost elements.
            The keys must be 'x', 'y', 'u', 'du', and 'total', and the
            values must be 2D arrays.

        Raises
        ------
        ValueError
            If the provided keys do not match the expected cost elements.
        """
        expected_keys = set(COST_ELEMENTS)
        provided_keys = set(new_values.keys())

        if provided_keys != expected_keys:
            msg = f'You must specify all cost elements: {expected_keys}.'
            raise ValueError()

        for element in COST_ELEMENTS:
            getattr(self, f'_{element}').arr = new_values[element]

    @property
    def df(self) -> pd.DataFrame:
        dfs = [
            self.x.df,
            self.y.df,
            self.u.df,
            self.du.df,
            self.total.df,
        ]
        return reduce(
            lambda left, right: pd.merge(left, right, on='k'),
            dfs,
        )

    # class PredictionsStructure:
    #     k = ReadOnly[VariableViewDict]()
    #     t = ReadOnly[VariableViewDict]()
    #     x = ReadOnly[VariableViewDict]()
    #     z = ReadOnly[VariableViewDict]()
    #     u = ReadOnly[VariableViewDict]()
    #     costs = ReadOnly[CostStructure]()
    #     x_fine = ReadOnly[VariableViewDict]()
    #     z_fine = ReadOnly[VariableViewDict]()

    #     def __init__(self, instance: 'ModelABC') -> None:
    #         self._time_step: IntType = instance.k
    #         # self._full_data: PredictionsData = instance.data.predictions_full
    #         self._x_vars: list[str] = instance.x_vars
    #         self._z_vars: list[str] = instance.z_vars
    #         self._u_vars: list[str] = instance.u_vars
    #         self._p_vars: list[str] = instance.p_vars
    #         self._q_vars: list[str] = instance.q_vars

    #         # Initialize each prediction element with its corresponding data
    #         for element in [*PREDICTION_ELEMENTS, 'x_fine', 'z_fine']:
    #             if element in ['x', 'x_fine']:
    #                 columns = [f'{name_}' for name_ in self._x_vars]
    #                 indexes = list(np.arange(len(self._x_vars)))
    #             elif element in ['z', 'z_fine']:
    #                 columns = [f'{name_}' for name_ in self._z_vars]
    #                 indexes = list(np.arange(len(self._z_vars)))
    #             elif element == 'u':
    #                 columns = [f'{name_}' for name_ in self._u_vars]
    #                 indexes = list(np.arange(len(self._u_vars)))
    #             elif element == 'p':
    #                 columns = [f'{name_}' for name_ in self._p_vars]
    #                 indexes = list(np.arange(len(self._p_vars)))
    #             elif element == 'q':
    #                 columns = [f'{name_}' for name_ in self._q_vars]
    #                 indexes = list(np.arange(len(self._q_vars)))
    #             elif element == 't':
    #                 columns = ['t']
    #                 indexes = [0]
    #             elif element == 'k':
    #                 columns = ['k']
    #                 indexes = [0]
    #             else:
    #                 msg = f'Element {element} is not a valid prediction element.'
    #                 raise ValueError(msg)

    #             variable_data: PredDType = getattr(self._full_data, element)
    #             setattr(
    #                 self,
    #                 f'_{element}',
    #                 VariableViewDict(
    #                     k=self._time_step,
    #                     full_data=variable_data,
    #                     indexes=indexes,
    #                     columns=columns,
    #                 ),
    #             )

    #         # Prepare cost data dictionary from full_data
    #         cost_data = {
    #             element: getattr(self._full_data, f'cost_{element}')
    #             for element in COST_ELEMENTS
    #         }

    #         # Initialize the cost structure with the cost data
    #         self._costs = CostStructure(k=self._time_step, full_data=cost_data)

    # @property
    # def df(self) -> pd.DataFrame:
    #     """
    #     Return a pandas DataFrame containing the cost data of the predictions.

    #     The columns of the DataFrame are 'step', 'time', 'cost_x', 'cost_y',
    #     'cost_u', 'cost_du', and 'cost_total', which represent the current
    #     time step, the current time, the cost of the states, the cost of the
    #     outputs, the cost of the inputs, the cost of the input differences,
    #     and the total cost, respectively.

    #     Returns
    #     -------
    #     df_cost : pd.DataFrame
    #         The DataFrame containing the cost data of the predictions.
    #     """

    #     dfs = [self.t.df, self.x.df, self.u.df, self.costs.df]

    #     return reduce(
    #         lambda left, right: pd.merge(left, right, on='k'),
    #         dfs,
    #     )

    # @property
    # def df_tux(self) -> pd.DataFrame:
    #     """
    #     Return a pandas DataFrame containing the cost data of the predictions.

    #     The columns of the DataFrame are 'step', 'time', 'cost_x', 'cost_y',
    #     'cost_u', 'cost_du', and 'cost_total', which represent the current
    #     time step, the current time, the cost of the states, the cost of the
    #     outputs, the cost of the inputs, the cost of the input differences,
    #     and the total cost, respectively.

    #     Returns
    #     -------
    #     df_cost : pd.DataFrame
    #         The DataFrame containing the cost data of the predictions.
    #     """

    #     dfs = [self.t.df, self.u.df, self.x.df, self.costs.df]

    #     return reduce(
    #         lambda left, right: pd.merge(left, right, on='k'),
    #         dfs,
    #     )

    # def get_var(self, var: str, is_fine: bool = False) -> VariableViewDict:
    #     if var in self._x_vars:
    #         indexes = [self._x_vars.index(var)]
    #         data = self._full_data.x if not is_fine else self._full_data.x_fine

    #     if var in self._z_vars:
    #         indexes = [self._z_vars.index(var)]
    #         data = self._full_data.z if not is_fine else self._full_data.z_fine

    #     if var in self._u_vars:
    #         indexes = [self._u_vars.index(var)]
    #         data = self._full_data.u

    #     return VariableViewDict(
    #         k=self._time_step,
    #         full_data=data,
    #         indexes=indexes,
    #         columns=[var],
    #     )

    #     msg = f'Variable {var} not found.'
    #     raise ValueError(msg)


# class SystemCallable:
#     def __set_name__(self, owner: type, name: str) -> None:
#         # Store the name of the descriptor
#         self._name = name

#     def __get__(
#         self, instance: 'ModelABC', owner
#     ) -> Callable[[str], VariableView]:
#         # Retrieve the list of variable names for state spaces
#         self._instance = instance

#         return self.get_val

# def get_val(self, var_: str) -> VariableView:
#     if self._name in GlobalState.x_vars:
#         return VariableView(
#             k=GlobalState.k(),
#             full_data=self._instance.data.x_full,
#             indexes=[GlobalState.x_vars().index(var)],
#         )


class DataVariable:
    def __set_name__(self, owner, name: str) -> None:
        self._name = name

    def __set__(self, instance, value: Any) -> None:
        msg = f"Attribute '{self._name}' is read-only."
        raise AttributeError(msg)

    def __get__(self, instance, owner):
        var_name = self._name

        return instance.data.goals
        # return VariableView(
        #     k=instance.k,
        #     full_data=instance.data.goals,
        #     indexes=np.arange(),
        #     columns=columns,
        # )


class GoalsStructure:
    x = ReadOnly[VariableViewDict]()
    u = ReadOnly[VariableViewDict]()
    y = ReadOnly[VariableViewDict]()


class SystemVariable:
    def __set_name__(self, owner, name: str) -> None:
        self._name = name
        self._private_name = f'_{name}'

    def __set__(self, instance, value: Any) -> None:
        msg = f"Attribute '{self._name}' is read-only."
        raise AttributeError(msg)

    def __get__(self, instance, owner):
        var_name = self._name

        full_data, indexes, columns = _get_full_data_and_indexes(
            instance,
            ss_name=var_name,
            name=var_name,
        )
        return VariableView(
            k=GlobalState.k(),
            full_data=full_data,
            indexes=indexes,
            columns=columns,
        )


class StateStructure:
    def __init__(
        self,
        k: IntType,
        full_data: Array2D,
        indexes: list[IntType],
        columns: list[str],
    ) -> None:
        self._k = k
        self._full_data = full_data
        self._indexes = indexes
        self._columns = columns

    @property
    def vec(self):
        return VariableView(
            k=self._k,
            full_data=self._full_data,
            indexes=self._indexes,
            columns=self._columns,
        )

    @property
    def goal(self):
        pass

    @property
    def max(self):
        pass

    @property
    def min(self):
        pass


class SystemTime:
    def __set_name__(self, owner, name: str) -> None:
        """
        Set the name of the private variable.

        This method is automatically called when the instance variable is
        assigned to the class. The name of the private variable is set to
        `_{name}`, where `{name}` is the name of the instance variable.

        Parameters
        ----------
        owner : type
            The class that owns this instance variable.
        name : str
            The name of the instance variable.
        """
        self._name = name
        self._private_name = f'_{name}'

    def __get__(
        self,
        instance: 'ModelABC',
        owner: type['ModelABC'],
    ) -> VariableView:
        if self._name == 't':
            # The time variable is a special case
            return VariableView(
                k=GlobalState.k(),
                full_data=instance.data.t_full,
                indexes=[0],  # The time variable only has one element
                columns=[self._name],
            )

        raise NotAvailableAttributeError(self._name)


# class SystemPredictions:
#     def __set_name__(self, owner, name: str) -> None:
#         """
#         Set the name of the private variable.

#         This method is automatically called when the instance variable is
#         assigned to the class. The name of the private variable is set to
#         `_{name}`, where `{name}` is the name of the instance variable.

#         Parameters
#         ----------
#         owner : type
#             The class that owns this instance variable.
#         name : str
#             The name of the instance variable.
#         """
#         self._name = name
#         self._private_name = f'_{name}'

#     def __get__(self, instance: 'ModelABC', owner) -> PredictionsStructure:
#         if self._name == 'predictions':
#             # The predictions variable is a special case
#             return PredictionsStructure(instance)
#         raise NotAvailableAttributeError(self._name)
