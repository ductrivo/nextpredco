from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import override

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from nextpredco.core import logger
from nextpredco.core.custom_types import Symbolic
from nextpredco.core.descriptors import ReadOnlyInt
from nextpredco.core.model import Model
from nextpredco.core.optimizer import IPOPT
from nextpredco.core.settings import (
    ControllerSettings,
    MPCSettings,
    PIDSettings,
)


class Controller(ABC):
    def __init__(
        self,
        settings: ControllerSettings | MPCSettings | PIDSettings,
        model: Model | None = None,
        optimizer: IPOPT | None = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._settings = settings
        self._settings.dt = model.dt if settings.dt == -1 else settings.dt

        self._n_ctrl = round(self._settings.dt / self._model.dt)

    @abstractmethod
    def make_step(self) -> NDArray:
        pass

    @property
    def model(self) -> Model | None:
        return self._model

    @property
    def optimizer(self) -> IPOPT | None:
        return self._optimizer


class PID(Controller):
    def __init__(
        self,
        pid_settings: PIDSettings,
    ):
        super().__init__(pid_settings)
        self._kp = pid_settings.kp
        self._ki = pid_settings.ki
        self._kd = pid_settings.kd

    @override
    def make_step(self):
        pass

    def auto_tuning(self):
        pass


class MPC(Controller):
    n_pred = ReadOnlyInt()

    def __init__(
        self,
        settings: MPCSettings,
        model: Model,
        optimizer: IPOPT | None,
    ):
        super().__init__(settings, model, optimizer)

        self._n_pred = settings.n_pred
        self._weights = {
            'x': settings.weight_x[0],
            'y': settings.weight_y[0],
            'u': settings.weight_u[0],
            'du': settings.weight_du[0],
        }
        x_pred, u_pred = self._make_prediction()
        cost = self.compute_cost(x_pred, u_pred)

    def _make_prediction(self):
        return self._predict_single_shooting()

    def compute_cost(
        self, x_pred: Symbolic | NDArray, u_pred: Symbolic | NDArray
    ) -> Symbolic | NDArray:
        # TODO: Normalize the errors
        costs = {}
        costs['x'] = self._compute_cost_ingredients(
            arr=x_pred,
            weight=self._weights['x'],
            goal=self._model.x.goal.val,
        )

        costs['u'] = self._compute_cost_ingredients(
            arr=u_pred,
            weight=self._weights['u'],
            goal=self._model.u.goal.val,
        )

        costs['y'] = self._compute_cost_ingredients(
            arr=self._model.get_y(x=x_pred),
            weight=self._weights['y'],
            goal=self._model.y.goal.val,
        )

        costs['du'] = self._compute_cost_ingredients(
            arr=self._model.get_du(
                u_arr=u_pred,
                u_prev=self._model.u.goal.val,
            ),
            weight=self._weights['du'],
        )
        return ca.sum1(ca.vcat(costs.values()))

    def get_constraints(self):
        pass

    def _compute_cost_ingredients(
        self,
        arr: Symbolic | NDArray,
        weight: NDArray | float,
        goal: Symbolic | NDArray | None = None,
    ) -> Symbolic:
        err = (
            self._compute_error(arr=arr, goal=goal)
            if goal is not None
            else arr
        )
        cost_arr = err.T @ weight @ err

        if isinstance(cost_arr, Symbolic):
            return ca.trace(cost_arr)
        return cost_arr.trace()

    def _compute_error(
        self, arr: Symbolic | NDArray, goal: Symbolic | NDArray
    ) -> Symbolic:
        if self._settings.normalizing:
            if isinstance(goal, Symbolic):
                logger.warning(
                    'Goal of type %s. Normalizing is disabled.',
                    Symbolic,
                )
            elif isinstance(goal, np.ndarray) and any(np.isclose(goal, 0)):
                logger.warning(
                    'Goal has zero element(s). Normalizing is disabled.'
                )
                return arr - goal

            return arr / goal - 1
        return arr - goal

    def _predict_single_shooting(self) -> Symbolic:
        # TODO: check if algebraic equations are present
        x0 = Symbolic.sym('__x0', self._model.n('x'), 1)
        z0 = Symbolic.sym('__z0', 0)
        u_pred = Symbolic.sym('__u_pred', self._model.n('u'), self._n_pred)
        x_pred = self._integrate_euler(
            h=self._settings.dt,
            x0=x0,
            z0=z0,
            f_func=self._model._transient_funcs['f'],
            u_pred=u_pred,
        )

        return x_pred, u_pred

    def _integrate_euler(
        self,
        h: float,
        x0: Symbolic,
        z0: Symbolic,
        f_func: ca.Function,
        u_pred: Symbolic,
    ) -> Symbolic:
        x_pred: ArrayLike = []
        x = x0
        for k in range(u_pred.shape[1]):
            upq = self._model.get_upq(u=u_pred[:, k])
            x += h * f_func(x0, z0, upq)
            x_pred.append(copy(x))

        return ca.hcat(x_pred)

    # def _predict_multiple_shooting(self) -> Symbolic:
    #     h = self._settings.dt
    #     funcs

    @override
    def make_step(self):
        pass
