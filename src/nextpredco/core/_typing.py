from typing import TypeVar

import numpy as np
from casadi import SX as Symbolic
from numpy.typing import NDArray

type SourceType = list[str] | tuple[str, ...]
type TgridType = list[float | int] | NDArray
type IntType = int | np.int_
type FloatType = float | np.float64 | np.float32
type Array2D = NDArray
type PredDType = dict[int, Array2D]
ArrayType = TypeVar('ArrayType', Symbolic, NDArray)
