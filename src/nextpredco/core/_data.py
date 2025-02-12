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
    k_clock: IntType
    k_max: IntType
    k_clock_max: IntType

    t_full: Array2D
    t_clock_full: Array2D
    x_est_full: Array2D
    z_est_full: Array2D
    upq_est_full: Array2D

    x_goal_full: Array2D = field(default_factory=lambda: np.array([[]]))
    z_goal_full: Array2D = field(default_factory=lambda: np.array([[]]))
    upq_goal_full: Array2D = field(default_factory=lambda: np.array([[]]))

    x_act_full: Array2D = field(default_factory=lambda: np.array([[]]))
    z_act_full: Array2D = field(default_factory=lambda: np.array([[]]))
    upq_act_full: Array2D = field(default_factory=lambda: np.array([[]]))

    x_meas_full: Array2D = field(default_factory=lambda: np.array([[]]))
    z_meas_full: Array2D = field(default_factory=lambda: np.array([[]]))
    upq_meas_full: Array2D = field(default_factory=lambda: np.array([[]]))

    x_filt_full: Array2D = field(default_factory=lambda: np.array([[]]))
    z_filt_full: Array2D = field(default_factory=lambda: np.array([[]]))
    upq_filt_full: Array2D = field(default_factory=lambda: np.array([[]]))

    x_goal_clock_full: Array2D = field(default_factory=lambda: np.array([[]]))
    z_goal_clock_full: Array2D = field(default_factory=lambda: np.array([[]]))
    upq_goal_clock_full: Array2D = field(
        default_factory=lambda: np.array([[]])
    )

    x_act_clock_full: Array2D = field(default_factory=lambda: np.array([[]]))
    z_act_clock_full: Array2D = field(default_factory=lambda: np.array([[]]))
    upq_act_clock_full: Array2D = field(default_factory=lambda: np.array([[]]))

    x_est_clock_full: Array2D = field(default_factory=lambda: np.array([[]]))
    z_est_clock_full: Array2D = field(default_factory=lambda: np.array([[]]))
    upq_est_clock_full: Array2D = field(default_factory=lambda: np.array([[]]))

    x_meas_clock_full: Array2D = field(default_factory=lambda: np.array([[]]))
    z_meas_clock_full: Array2D = field(default_factory=lambda: np.array([[]]))
    upq_meas_clock_full: Array2D = field(
        default_factory=lambda: np.array([[]])
    )

    x_filt_clock_full: Array2D = field(default_factory=lambda: np.array([[]]))
    z_filt_clock_full: Array2D = field(default_factory=lambda: np.array([[]]))
    upq_filt_clock_full: Array2D = field(
        default_factory=lambda: np.array([[]])
    )

    # MPC predictions
    predictions_full: PredictionsData = field(default_factory=PredictionsData)

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
