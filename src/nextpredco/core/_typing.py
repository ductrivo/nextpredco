from collections import OrderedDict
from typing import Any, TypeVar

import numpy as np
from casadi import MX as Symbolic
from numpy.typing import NDArray

type SourceType = list[str] | tuple[str, ...]
type TgridType = list[float | int] | NDArray
type IntType = int | np.int_
type FloatType = float | np.float64 | np.float32
type Array2D = NDArray
type PredDType = OrderedDict[IntType, Array2D]
ArrayType = TypeVar('ArrayType', Symbolic, NDArray)


def isIntType(var: IntType | Any) -> bool:
    return isinstance(var, int | np.int_)


def isArray2D(var: Array2D | Any) -> bool:
    return isinstance(var, np.ndarray) and var.ndim == 2
