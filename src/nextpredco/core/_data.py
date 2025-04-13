import sys
from collections import OrderedDict as OrderedDict
from dataclasses import dataclass, field

import numpy as np

from nextpredco.core._typing import (
    Array2D,
    FloatType,
    IntType,
    PredDType,
)

type PredictionsType = OrderedDict[IntType, PredictionsData]


@dataclass
class PredictionsData:
    k: PredDType = field(default_factory=OrderedDict)
    t: PredDType = field(default_factory=OrderedDict)
    x: PredDType = field(default_factory=OrderedDict)
    z: PredDType = field(default_factory=OrderedDict)
    u: PredDType = field(default_factory=OrderedDict)
    x_fine: PredDType = field(default_factory=OrderedDict)
    z_fine: PredDType = field(default_factory=OrderedDict)
    cost_x: PredDType = field(default_factory=OrderedDict)
    cost_y: PredDType = field(default_factory=OrderedDict)
    cost_u: PredDType = field(default_factory=OrderedDict)
    cost_du: PredDType = field(default_factory=OrderedDict)
    cost_total: PredDType = field(default_factory=OrderedDict)


@dataclass
class ModelData:
    k: IntType

    k_max: IntType

    t_full: Array2D

    x_full: Array2D
    z_full: Array2D
    upq_full: Array2D

    x_min: OrderedDict[IntType, Array2D] = field(default_factory=OrderedDict)
    z_min: OrderedDict[IntType, Array2D] = field(default_factory=OrderedDict)
    upq_min: OrderedDict[IntType, Array2D] = field(default_factory=OrderedDict)

    x_max: OrderedDict[IntType, Array2D] = field(default_factory=OrderedDict)
    z_max: OrderedDict[IntType, Array2D] = field(default_factory=OrderedDict)
    upq_max: OrderedDict[IntType, Array2D] = field(default_factory=OrderedDict)

    @property
    def size(self) -> IntType:
        def get_size(obj) -> int:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            if isinstance(obj, dict):
                size = sys.getsizeof(obj)
                for key, value in obj.items():
                    size += get_size(key)
                    size += get_size(value)
                return size
            if isinstance(obj, list):
                size = sys.getsizeof(obj)
                for item in obj:
                    size += get_size(item)
                return size
            return sys.getsizeof(obj)

        total_size = 0
        for field_name, field_value in self.__dict__.items():
            total_size += get_size(field_value)
        return total_size
