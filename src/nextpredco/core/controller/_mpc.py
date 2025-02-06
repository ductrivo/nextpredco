from copy import copy
from typing import override

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from nextpredco.core import Symbolic
from nextpredco.core._logger import logger
from nextpredco.core.controller import ControllerABC
from nextpredco.core.model import Model
from nextpredco.core.model._descriptors import ReadOnlyInt
from nextpredco.core.settings import (
    IntegratorSettings,
    MPCSettings,
    OptimizerSettings,
)


class MPC(ControllerABC):
    n_pred = ReadOnlyInt()

    def __init__(
        self,
        settings: MPCSettings,
        model: Model,
        optimizer_settings: OptimizerSettings | None = None,
        integrator_settings: IntegratorSettings | None = None,
    ):
        super().__init__(
            settings, model, optimizer_settings, integrator_settings
        )
        # To ensure that auto completion only shows MPCSettings attributes
        self._settings: MPCSettings

        self._n_pred = settings.n_pred
        self._weights = {
            'x': settings.weight_x[0],
            'y': settings.weight_y[0],
            'u': settings.weight_u[0],
            'du': settings.weight_du[0],
        }

        # self._nlp_solver = self._create_optimization_problem()
        # self._nlp_solver(
        #     x0=np.repeat(self._model.u.goal.val, self._n_pred, axis=0),
        #     p=np.vstack(
        #         [
        #             self._model.x.est.val,
        #             self._model.z.est.val,
        #             self._model.p.est.val,
        #             self._model.q.est.val,
        #         ]
        #     ),
        # )

    def _create_optimization_problem(self):
        return self._predict_single_shooting()

    def compute_cost(
        self,
        x_pred: Symbolic | NDArray,
        u_pred: Symbolic | NDArray,
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
                u_last=self._model.u.goal.val,
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

    def _predict_single_shooting(self) -> ca.Function:
        # TODO: check if algebraic equations are present
        x0 = Symbolic.sym('__x0', self._model.n('x'), 1)
        z0 = Symbolic.sym('__z0', 0)
        p0 = Symbolic.sym('__p0', self._model.n('p'), 1)
        q0 = Symbolic.sym('__q0', self._model.n('q'), 1)

        u_pred_vec: Symbolic = Symbolic.sym(
            'u_pred', self._model.n('u') * self._n_pred, 1
        )

        # Note: casadi use column-major order,
        # numpy use row-major order
        u_pred = u_pred_vec.reshape((self._model.n('u'), self._n_pred))
        # Make prediction

        upq_arr_ = []
        for k in range(self._n_pred):
            upq_ = self._model.get_upq(u=u_pred[:, k], p=p0, q=q0)
            upq_arr_.append(upq_)

        upq_arr = ca.hcat(upq_arr_)

        _, _, x_pred, _ = self._integrator.integrate(
            x0=x0,
            z0=z0,
            upq_arr=upq_arr,
        )

        # Compute cost
        cost = self.compute_cost(x_pred, u_pred)

        # Create nlp info
        nlp = {
            'x': u_pred_vec,
            'f': cost,
            'p': ca.vcat([x0, z0, p0, q0]),
        }

        # Create nlp solver
        nlp_solver: ca.Function = ca.nlpsol('nlp_solver', 'ipopt', nlp)
        return nlp_solver

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

    @override
    def make_step(self):
        pass
