from collections import OrderedDict as OrderDict
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rich.pretty import pretty_repr as pretty_repr

from nextpredco.core._consts import (
    COST_ELEMENTS,
    PREDICTION_ELEMENTS,
    SS_VARS_DB,
    SS_VARS_SECONDARY,
)
from nextpredco.core._data import PredictionsData
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

ReadOnlyType = TypeVar(
    'ReadOnlyType',
    SourceType,
    list[str],
    IntType,
    FloatType,
    'VariableView',
    'VariableViewDict',
    'CostStructure',
)


class ReadOnly2(Generic[ReadOnlyType]):
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
            return getattr(instance._data, self._name)
        msg = f'Invalid data container: {data_container_name}'
        raise AttributeError(msg)


class VariableView:
    def __init__(
        self,
        k: IntType,
        full_data: Array2D,
        idx_list: list[IntType],
    ) -> None:
        """Initialize the variable view.

        Parameters
        ----------
        k : IntType
            The current time step.
        full_data : Array2D
            The full data array.
        idx_list : list[IntType]
            The list of indices.
        """
        self._k = k
        self._full_data = full_data
        self._indexes = idx_list

    @property
    def val(self) -> Array2D:
        """Get the current value of the variable view.

        Returns
        -------
        Array2D
            The current value of the variable view.
        """
        return self.get_val(self._k)

    @val.setter
    def val(self, val: Array2D) -> None:
        """
        Set the current value of the variable view.

        Parameters
        ----------
        val : Array2D
            The value to set.
        """
        self.set_val(self._k, val)

    @property
    def last(self) -> Array2D:
        """Get the previous value of the variable view.

        Returns
        -------
        Array2D
            The previous value of the variable view.
        """
        return self.get_val(self._k - 1)

    @last.setter
    def last(self, val: Array2D) -> None:
        """
        Set the previous value of the variable view.

        Parameters
        ----------
        val : Array2D
            The value to set.
        """
        self.set_val(self._k - 1, val)

    @property
    def hist(self) -> Array2D:
        """Get the historical values of the variable view.

        Returns
        -------
        Array2D
            The historical values of the variable view from the beginning
            up to the current time step.
        """

        return self.get_hist(0, self._k)

    @hist.setter
    def hist(self, val: Array2D) -> None:
        """
        Set the historical values of the variable view.

        Parameters
        ----------
        val : Array2D
            The values to set.
        """
        self.set_hist(0, self._k, val)

    @property
    def full(self) -> Array2D:
        """
        Get the full array of the variable view.

        Returns
        -------
        Array2D
            The full array of the variable view.
        """
        return self._full_data

    @full.setter
    def full(self, val: Array2D) -> None:
        """
        Set the full array of the variable view.

        Parameters
        ----------
        val : Array2D
            The values to set.
        """
        self._full_data = val

    def get_val(self, k: IntType) -> Array2D:
        """
        Retrieve the value of the variable view at a specified time step.

        Parameters
        ----------
        k : IntType
            The time step for which the value is to be retrieved.

        Returns
        -------
        Array2D
            The value of the variable view at the specified time step.
        """

        return self.get_hist(k0=k, k1=k)

    def set_val(self, k: IntType, val: Array2D) -> None:
        """
        Set the value of the variable view at a specified time step.

        Parameters
        ----------
        k : IntType
            The time step at which the value is to be set.
        val : Array2D
            The value to set.
        """
        self.set_hist(k0=k, k1=k, val=val)

    def get_hist(
        self,
        k0: IntType,
        k1: IntType,
    ) -> Array2D:
        """
        Retrieve the historical values of the variable view.

        Parameters
        ----------
        k0 : IntType
            The initial time step from which the historical values are
            to be retrieved.
        k1 : IntType
            The final time step up to which the historical values are
            to be retrieved.

        Returns
        -------
        Array2D
            The historical values of the variable view from the initial
            to the final time step.

        Raises
        ------
        ValueError
            If k0 is less than 0.
        """
        if k0 < 0:
            msg = f'k0 must be greater than or equal to 0. You entered {k0}.'
            raise ValueError(msg)

        # If the full data array is empty, return an empty array
        if len(self._indexes) == 0:
            return np.array([[]]).T

        # If the full data array is a view of the entire array, return a slice
        if len(self._indexes) == self._full_data.shape[0]:
            return self._full_data[:, k0 : (k1 + 1)]

        # Otherwise, return a slice of the view
        return self._full_data[self._indexes, k0 : (k1 + 1)]

    def set_hist(
        self,
        k0: IntType,
        k1: IntType,
        val: Array2D,
    ) -> None:
        """
        Set the historical values of the variable view.

        Parameters
        ----------
        k0 : IntType
            The initial time step from which the historical values are
            to be set.
        k1 : IntType
            The final time step up to which the historical values are
            to be set.
        val : Array2D
            The values to set.

        Raises
        ------
        TypeError
            If the value is not a 2D array.
        ValueError
            If k0 is less than 0.
        """
        # Check if the value is a 2D array
        if not isArray2D(val):
            msg = f'The value must be a 2D array. You entered {type(val)}.'
            raise TypeError(msg)

        # Check if k0 is valid
        if k0 < 0:
            msg = f'k0 must be greater than or equal to 0. You entered {k0}.'
            raise ValueError(msg)

        # If the full data array is the same as the variable view's data array,
        # the set the values directly
        if len(self._indexes) == self._full_data.shape[0]:
            self._full_data[:, k0 : (k1 + 1)] = val
        # Otherwise, set the values to the corresponding view of the variable
        # and view's data array
        else:
            self._full_data[self._indexes, k0 : (k1 + 1)] = val


class VariableViewDict:
    """
    A class for representing a variable view dictionary.

    Parameters
    ----------
    k : IntType
        The current time step.
    full_data : PredictionsData | Array2D
        The full data array.
    indexes : list[IntType], optional
        The list of indices. If not provided, the full data array is used.
    """

    def __init__(
        self,
        k: IntType,
        full_data: PredDType,
        indexes: list[IntType] | None = None,
    ) -> None:
        """
        Initialize the variable view dictionary.

        Parameters
        ----------
        k : IntType
            The current time step.
        full_data : PredictionsData | Array2D
            The full data array.
        indexes : list[IntType], optional
            The list of indices. If not provided, the full data array is used.
        """
        self._k = k
        self._data_full = full_data
        self._indexes = indexes

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

        if k == -1:
            ordered = OrderDict(self._data_full)
            input(next(reversed(ordered.values())))
            return next(reversed(ordered.values()))

        if k not in self._data_full:
            msg = f'The time step {k} is not in the data.'
            raise ValueError(msg)

        arr = self._data_full[k]

        # If there are no indexes, return the entire array
        if self._indexes is None:
            return arr

        # If there are no indexes, return an empty array
        if len(self._indexes) == 0:
            return np.array([[]])

        # If the indexes span the entire array, return the entire array
        # if len(self._indexes) == arr.shape[0]:
        #     return arr

        # Otherwise, return the selected elements
        return arr[self._indexes, :]

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
        self._data_full[k] = val

    @property
    def vstack(self) -> NDArray:
        """
        Stack the values of the full_data dictionary vertically.

        Returns
        -------
        NDArray
            The stacked values.
        """
        data = [np.vstack(list(self._data_full.values()))]
        return np.hstack(data)


class StateSpaceStructure:
    """
    A class to hold the state space structure for a model.

    It contains the full data arrays for each source, as well as the index lists
    for each source. It also contains the current time step and the time step
    in clock time.

    Attributes
    ----------
    goal : VariableView
        The state space structure for the goal data.
    act : VariableView
        The state space structure for the actual data.
    est : VariableView
        The state space structure for the estimated data.
    meas : VariableView
        The state space structure for the measurement data.
    filt : VariableView
        The state space structure for the filtered data.
    preds : dict[str, VariableView]
        A dictionary with the state space structures for the predicted data
        for each source.
    goal_clock : VariableView
        The state space structure for the goal data in clock time.
    act_clock : VariableView
        The state space structure for the actual data in clock time.
    est_clock : VariableView
        The state space structure for the estimated data in clock time.
    meas_clock : VariableView
        The state space structure for the measurement data in clock time.
    filt_clock : VariableView
        The state space structure for the filtered data in clock time.
    """

    goal = ReadOnly2[VariableView]()
    act = ReadOnly2[VariableView]()
    est = ReadOnly2[VariableView]()
    meas = ReadOnly2[VariableView]()
    filt = ReadOnly2[VariableView]()
    preds = ReadOnly2[VariableViewDict]()

    goal_clock = ReadOnly2[VariableView]()
    act_clock = ReadOnly2[VariableView]()
    est_clock = ReadOnly2[VariableView]()
    meas_clock = ReadOnly2[VariableView]()
    filt_clock = ReadOnly2[VariableView]()

    def __init__(
        self,
        k: IntType,
        k_clock: IntType,
        data_fulls: dict[str, Array2D],
        idx_lists: dict[str, list[IntType]],
    ) -> None:
        """
        Initialize the state space structure.

        Parameters
        ----------
        k : IntType
            The current time step.
        k_clock : IntType
            The current time step in clock time.
        data_fulls : dict[str, Array2D]
            A dictionary with the full data arrays for each source.
        idx_lists : dict[str, list[IntType]]
            A dictionary with the index lists for each source.
        """
        # Iterate over the sources and create the VariableView instances
        for source, full_data in data_fulls.items():
            # Determine the time step
            k_ = k_clock if 'clock' in source else k

            # Create the VariableView instance
            idx_list = idx_lists[source]
            variable_view = VariableView(k_, full_data, idx_list)

            # Set the attribute
            attr_name = f'_{source}'
            setattr(self, attr_name, variable_view)


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

    x = ReadOnly2[VariableViewDict]()
    y = ReadOnly2[VariableViewDict]()
    u = ReadOnly2[VariableViewDict]()
    du = ReadOnly2[VariableViewDict]()
    total = ReadOnly2[VariableViewDict]()

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
        self._data_full = full_data
        for cost_element in COST_ELEMENTS:
            variable_view_dict = VariableViewDict(
                k=k, full_data=full_data[cost_element]
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


class PredictionsStructure:
    k = ReadOnly2[VariableViewDict]()
    t = ReadOnly2[VariableViewDict]()
    x = ReadOnly2[VariableViewDict]()
    z = ReadOnly2[VariableViewDict]()
    u = ReadOnly2[VariableViewDict]()
    costs = ReadOnly2[CostStructure]()

    def __init__(self, k: IntType, full_data: PredictionsData) -> None:
        """
        Initialize the prediction structure.

        Parameters
        ----------
        k : IntType
            The number of steps in the prediction horizon.
        full_data : PredictionsData
            The full data of the predictions.

        The prediction structure is initialized with the given full data and
        the number of steps in the prediction horizon. The elements of the
        prediction structure are set to the corresponding elements of the
        full data.

        The cost elements are also initialized with the given full data.
        """
        # Initialize each prediction element with its corresponding data
        for element in PREDICTION_ELEMENTS:
            variable_data = getattr(full_data, element)
            setattr(
                self,
                f'_{element}',
                VariableViewDict(
                    k=k,
                    full_data=variable_data,
                ),
            )

        # Prepare cost data dictionary from full_data
        cost_data = {
            element: getattr(full_data, f'cost_{element}')
            for element in COST_ELEMENTS
        }

        # Initialize the cost structure with the cost data
        self._costs = CostStructure(k=k, full_data=cost_data)

    @property
    def df_cost(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame containing the cost data of the predictions.

        The columns of the DataFrame are 'step', 'time', 'cost_x', 'cost_y',
        'cost_u', 'cost_du', and 'cost_total', which represent the current
        time step, the current time, the cost of the states, the cost of the
        outputs, the cost of the inputs, the cost of the input differences,
        and the total cost, respectively.

        Returns
        -------
        df_cost : pd.DataFrame
            The DataFrame containing the cost data of the predictions.
        """
        cost_data = {
            'step': self.k.vstack[:, 0],
            'time': self.t.vstack[:, 0],
            'cost_x': self.costs.x.vstack[:, 0],
            'cost_y': self.costs.y.vstack[:, 0],
            'cost_u': self.costs.u.vstack[:, 0],
            'cost_du': self.costs.du.vstack[:, 0],
            'cost_total': self.costs.total.vstack[:, 0],
        }
        return pd.DataFrame(cost_data)


class SystemCallable:
    def __set_name__(self, owner: type, name: str) -> None:
        """
        Set the name of the descriptor.

        This method is automatically called when the descriptor is assigned
        to a class attribute. It sets the name of the descriptor to the name
        of the attribute it is assigned to.

        Parameters
        ----------
        owner : type
            The class that owns the attribute this descriptor is assigned to.
        name : str
            The name of the attribute this descriptor is assigned to.
        """
        # Store the name of the descriptor
        self._name = name

    def __get__(self, instance, owner) -> Callable[[str], StateSpaceStructure]:
        """
        Provide a callable to retrieve state space structures.

        This method returns a callable function that can be used to
        obtain the state space structure corresponding to a specified
        variable name from the instance of the model.

        Parameters
        ----------
        instance : Model
            The instance of the model.
        owner : type
            The class that owns this descriptor.

        Returns
        -------
        Callable[[str], StateSpaceStructure]
            A function that accepts a variable name and returns the
            associated state space structure.
        """
        # Retrieve the list of variable names for state spaces
        x_vars: list[str] = instance._settings.x_vars
        z_vars: list[str] = instance._settings.z_vars
        upq_vars: list[str] = instance._settings.upq_vars

        # Retrieve the available data sources from the model settings
        sources: SourceType = instance._settings.sources

        # Store instance and variable information for later use
        self._instance = instance
        self._x_vars = x_vars
        self._z_vars = z_vars
        self._upq_vars = upq_vars
        self._sources = sources

        # Return the function to get state space structures for given variable names
        return self.get_val

    def get_val(self, var_name: str) -> StateSpaceStructure:
        """
        Retrieve the state space structure for a specified variable name.

        Parameters
        ----------
        var_name : str
            The name of the variable for which the state space structure
            is to be retrieved.

        Returns
        -------
        StateSpaceStructure
            The state space structure associated with the given variable name.

        Raises
        ------
        NotAvailableAttributeError
            If the variable name is not recognized as part of the state space.
        """
        # Determine the state space name based on the variable name
        if var_name in self._x_vars:
            ss_name = 'x'
        elif var_name in self._z_vars:
            ss_name = 'z'
        elif var_name in self._upq_vars:
            ss_name = 'upq'
        else:
            raise NotAvailableAttributeError(var_name)

        # Initialize dictionaries to hold data and index lists for each source
        data: dict[str, Array2D] = {}
        idx_list: dict[str, list[IntType]] = {}

        # Iterate over each source to populate data and index lists
        for source in self._sources:
            data[source], idx_list[source] = _get_full_data_and_indexes(
                self._instance,
                ss_name=ss_name,
                name=var_name,
                source=source,
            )

        # Return a StateSpaceStructure with the retrieved data and index lists
        return StateSpaceStructure(
            self._instance._data.k,
            self._instance._data.k_clock,
            data,
            idx_list,
        )


class SystemVariable:
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

    def __set__(self, instance, value: Any) -> None:
        """
        Disallow direct assignment to this attribute.

        This attribute is intended to be read-only. Any attempt to assign a
        value to this attribute will result in an AttributeError.

        Raises
        ------
        AttributeError
            If an attempt is made to assign a value to this attribute.
        """
        msg = (
            f"Attribute '{self._name}' is read-only."
            f'{self._private_name} can be '
            'used to access the underlying data.'
        )
        raise AttributeError(msg)

    def __get__(
        self, instance, owner
    ) -> StateSpaceStructure | VariableView | PredictionsStructure:
        """
        Get the state space structure.

        This method is a getter for the instance variable. It returns the state
        space structure for the given variable name.

        Parameters
        ----------
        instance : Model
            The model instance.
        owner : type
            The class that owns this instance variable.

        Returns
        -------
        StateSpaceStructure | VariableView | PredictionsStructure
            The state space structure for the given variable name.
        """

        # Get the variables
        var_name = self._name
        var_sources = instance._settings.sources

        if self._name == 't':
            # The time variable is a special case
            return VariableView(
                k=instance.k,
                full_data=instance._data.t_full,
                idx_list=[0],  # The time variable only has one element
            )

        if self._name == 't_clock':
            # The clock time variable is a special case
            return VariableView(
                k=instance.k,
                full_data=instance._data.t_clock_full,
                idx_list=[0],  # The clock time variable only has one element
            )
        if self._name == 'predictions':
            # The predictions variable is a special case
            return PredictionsStructure(
                k=instance.k,
                full_data=instance._data.predictions_full,
            )

        # Get the full data and index lists
        arr_fulls = {}
        idx_lists = {}
        for source in var_sources:
            # Get the full data and index lists for each source
            arr_full, idx_list = _get_full_data_and_indexes(
                instance,
                ss_name=var_name,
                name=var_name,
                source=source,
            )
            arr_fulls[source] = arr_full
            idx_lists[source] = idx_list

        # Return the state space structure
        return StateSpaceStructure(
            instance.k,
            instance._data.k_clock,
            arr_fulls,
            idx_lists,
        )


def _get_full_data_and_indexes(
    instance,
    ss_name: str,
    name: str,
    source: str,
) -> tuple[Array2D, list[IntType]]:
    """Get the full data and index list for a state space variable.

    Parameters
    ----------
    instance : Model
        The model instance.
    ss_name : str
        The name of the state space (x, z, or upq).
    name : str
        The name of the variable.
    source : str
        The source of the variable (e.g., "est", "meas", etc.).

    Returns
    -------
    full_data : Array2D
        The full data array.
    idx_list : list[IntType]
        The index list.
    """

    # Define the variable names
    x_vars = instance._settings.x_vars
    z_vars = instance._settings.z_vars
    upq_vars = instance._settings.upq_vars

    # Get the full data
    if name in ['x', *x_vars, *SS_VARS_SECONDARY]:
        full_data = getattr(instance._data, f'x_{source}_full')
    elif name in ['z', *z_vars]:
        full_data = getattr(instance._data, f'z_{source}_full')
    elif name in ['u', 'p', 'q', 'upq', *upq_vars]:
        full_data = getattr(instance._data, f'upq_{source}_full')
    else:
        raise NotAvailableAttributeError(name)

    # Get the index list
    if name in SS_VARS_DB:
        idx_list = list(range(full_data.shape[0]))

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

    return full_data, idx_list
