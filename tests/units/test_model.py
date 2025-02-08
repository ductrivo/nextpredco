import logging

import numpy as np
import pytest
from numpy.typing import NDArray

from nextpredco.core._consts import (
    SS_VARS_DB,
    SS_VARS_SOURCES,
)
from nextpredco.core._logger import logger
from nextpredco.core.errors import DescriptorSetError, MemoryAddressError
from nextpredco.core.model import Model, SystemVariableView

logger.setLevel(logging.INFO)


@pytest.mark.parametrize('test_id', range(30))
def test_address_matching_full(test_id: int):
    """Test if the state space variables are correctly defined in the model.

    The descriptor should point to the same memory address as the
    attribute that stores the value of the variable.
    In other words, the descriptor must not create a new array.
    """
    # TODO: Test if errors are raised

    model = Model()
    k0, k = sorted(np.random.randint(0, model.k_max - 1, size=2))
    model._data.k = k
    for ss_var in SS_VARS_DB:
        for source in SS_VARS_SOURCES:
            for view in ['full', 'val', 'prev', 'hist']:
                # get the attribute names
                descriptor_name = f'model.{ss_var}.{source}.{view}'
                arr_name = f'model._{ss_var}_{source}_{view}'

                # get the attribute values
                arr_full = getattr(model, f'_{ss_var}_{source}_full')
                descriptor_source: SystemVariableView = getattr(
                    getattr(model, ss_var),
                    source,
                )
                descriptor_view: NDArray = getattr(
                    descriptor_source,
                    view,
                )

                # get the array view based on the view type
                if view == 'full':
                    arr_view = arr_full
                elif view == 'val':
                    arr_view = arr_full[:, model.k, None]
                elif view == 'prev':
                    arr_view = arr_full[:, model.k - 1, None]
                elif view == 'hist':
                    arr_view = arr_full[:, : (model.k + 1)]

                # NOTE: If the array is empty, the view will be None
                arr_view = None if arr_view.size == 0 else arr_view

                # check if the arrays share the same memory address
                if not are_identical_arrays(descriptor_view, arr_view):
                    raise MemoryAddressError(
                        name1=descriptor_name,
                        name2=arr_name,
                    )

                # check set_val of the descriptor
                if view == 'val' and arr_view is not None:
                    check_set_val(
                        k=model.k,
                        arr_view=arr_view,
                        descriptor_source=descriptor_source,
                        descriptor_view=descriptor_view,
                        name=descriptor_name,
                    )

                # check set_hist of the descriptor
                elif view == 'hist' and arr_view is not None:
                    arr_view_hist = descriptor_source.get_hist(k0, k)
                    check_set_get_hist(
                        k0=k0,
                        k=model.k,
                        arr_view=arr_view_hist,
                        descriptor_source=descriptor_source,
                        descriptor_view=descriptor_view,
                        name=descriptor_name,
                    )


def check_set_val(
    k: int,
    arr_view: NDArray,
    descriptor_source: SystemVariableView,
    descriptor_view: NDArray,
    name: str,
):
    rand_val = np.random.randint(
        low=-1000,
        high=1000,
        size=arr_view.shape,
    )
    descriptor_source.set_val(k, rand_val)

    if not np.array_equal(descriptor_view, rand_val):
        raise DescriptorSetError(
            name=name,
            rand_val=rand_val,
        )


def check_set_get_hist(
    k0: int,
    k: int,
    arr_view: NDArray,
    descriptor_source: SystemVariableView,
    descriptor_view: NDArray,
    name: str,
):
    rand_val = np.random.randint(
        low=-1000,
        high=1000,
        size=arr_view.shape,
    )
    descriptor_source.set_hist(k0, k, rand_val)

    if not (
        np.array_equal(descriptor_view, rand_val)
        or np.array_equal(arr_view, rand_val)
    ):
        raise DescriptorSetError(
            name=name,
            rand_val=rand_val,
        )


def are_identical_arrays(arr1: NDArray, arr2: NDArray) -> bool:
    """
    Check if two NumPy arrays share the same memory address
    and have the same shape.

    Parameters:
    arr1: First NumPy array.
    arr2: Second NumPy array.

    Returns:
    bool: True if the arrays share the same memory address and shape,
    False otherwise.
    """
    # print(f'arr1: {arr1}')
    # print(f'arr2: {arr2}')
    if (arr1 is None) and (arr2 is None):
        return True

    if (arr1 is None) or (arr2 is None):
        return False

    address1 = arr1.__array_interface__['data'][0]
    address2 = arr2.__array_interface__['data'][0]

    # print(f'address1: {address1}')
    # print(f'address2: {address2}')
    # print(f'arr1.shape: {arr1.shape}')
    # print(f'arr2.shape: {arr2.shape}')
    # print(f'arr1.strides: {arr1.strides}')
    # print(f'arr2.strides: {arr2.strides}')
    return (
        address1 == address2
        and arr1.shape == arr2.shape
        and arr1.strides == arr2.strides
    )
