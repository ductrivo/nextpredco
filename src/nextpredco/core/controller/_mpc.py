from copy import copy
from typing import override

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from nextpredco.core._logger import logger
from nextpredco.core._typing import ArrayType, Symbolic
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

        self._create_data_preds_in_model()

        # TODO Verify when need to re-create the nlp solver
        self._nlp_solver: ca.Function
        self._nlp_funcs: dict[str, ca.Function]
        self._nlp_solver, self._nlp_funcs = self._create_nlp_solver()

    def _create_data_preds_in_model(self):
        k_stamps = np.arange(
            start=0,
            stop=self._model.k_max + self._n_pred,
            step=self._n_pred,
            dtype=int,
        )

        k_preds = {
            k: k + np.arange(self.n_pred, dtype=int) + 1 for k in k_stamps
        }

        u_preds = {
            k: np.zeros((self.model.n('u'), self.n_pred)) for k in k_stamps
        }
        x_preds = {
            k: np.zeros((self.model.n('x'), self.n_pred)) for k in k_stamps
        }

        self._model.settings.sources.append('preds')
        self._model.create_data_preds_full(
            k_preds_full=k_preds,
            u_preds_full=u_preds,
            x_preds_full=x_preds,
        )

    def _create_nlp_solver(self):
        return self._single_shooting()

    def compute_costs(
        self,
        x_preds: ArrayType,
        u_preds: ArrayType,
    ) -> dict[str, ArrayType]:
        # TODO: Normalize the errors
        costs = {}
        costs['x'] = self._compute_cost_ingredients(
            arr=x_preds,
            weight=self._weights['x'],
            goal=self._model.x.goal.val,
        )

        costs['u'] = self._compute_cost_ingredients(
            arr=u_preds,
            weight=self._weights['u'],
            goal=self._model.u.goal.val,
        )

        costs['y'] = self._compute_cost_ingredients(
            arr=self._model.get_y(x=x_preds),
            weight=self._weights['y'],
            goal=self._model.y.goal.val,
        )

        costs['du'] = self._compute_cost_ingredients(
            arr=self._model.get_du(
                u_arr=u_preds,
                u_last=self._model.u.goal.val,
            ),
            weight=self._weights['du'],
        )
        total = ca.sum1(ca.vcat(costs.values()))
        costs['total'] = total
        return costs

    def get_constraints(self):
        pass

    def _compute_cost_ingredients(
        self,
        arr: ArrayType,
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
        self, arr: ArrayType, goal: Symbolic | NDArray
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

    def _single_shooting(self) -> ca.Function:
        # TODO: check if algebraic equations are present
        x0 = Symbolic.sym('__x0', self._model.n('x'), 1)
        z0 = Symbolic.sym('__z0', 0)
        p0 = Symbolic.sym('__p0', self._model.n('p'), 1)
        q0 = Symbolic.sym('__q0', self._model.n('q'), 1)

        u_pred_vec: Symbolic = Symbolic.sym(
            'u_preds', self._model.n('u') * self._n_pred, 1
        )

        # Note: casadi use column-major order,
        # numpy use row-major order
        u_preds = u_pred_vec.reshape((self._model.n('u'), self._n_pred))
        upq_arr_ = []
        for k in range(self._n_pred):
            upq_ = self.model.get_upq(u=u_preds[:, k], p=p0, q=q0)
            upq_arr_.append(upq_)
        upq_arr = ca.hcat(upq_arr_)

        _, _, x_preds, _ = self._integrator.integrate(
            x0=x0,
            z0=z0,
            upq_arr=upq_arr,
            t_grid=np.arange(self._n_pred + 1) * self._settings.dt,
        )

        # Compute cost
        costs = self.compute_costs(x_preds, u_preds)

        # Create nlp info
        nlp = {
            'x': u_pred_vec,
            'f': costs['total'],
            'p': ca.vcat([x0, z0, p0, q0]),
        }

        nlp_funcs = {
            'x_preds': ca.Function(
                'x_preds_func',
                [u_pred_vec, ca.vcat([x0, z0, p0, q0])],
                [x_preds],
            )
        }
        for key, val in costs.items():
            nlp_funcs[key] = ca.Function(
                f'{key}_func',
                [u_pred_vec, ca.vcat([x0, z0, p0, q0])],
                [val],
            )

        # Create nlp solver
        nlp_solver: ca.Function = ca.nlpsol('nlp_solver', 'ipopt', nlp)
        return nlp_solver, nlp_funcs

    def _integrate_euler(
        self,
        h: float,
        x0: Symbolic,
        z0: Symbolic,
        f_func: ca.Function,
        u_preds: Symbolic,
    ) -> Symbolic:
        x_preds: ArrayLike = []
        x = x0
        for k in range(u_preds.shape[1]):
            upq = self._model.get_upq(u=u_preds[:, k])
            x += h * f_func(x0, z0, upq)
            x_preds.append(copy(x))

        return ca.hcat(x_preds)

    @override
    def make_step(self):
        u_guess = np.repeat(self._model.u.goal.val, self._n_pred)
        params = np.vstack(
            [
                self._model.x.est.val,
                self._model.z.est.val,
                self._model.p.est.val,
                self._model.q.est.val,
            ]
        )
        sol_ = self._nlp_solver(x0=u_guess, p=params)

        sol: dict[str, NDArray]
        sol = {key: val.full() for key, val in sol_.items()}

        self._model.u.preds.horizon = sol['x'].reshape(
            (-1, self._n_pred), order='F'
        )

        self._model.x.preds.horizon = self._nlp_funcs['x_preds'](
            sol['x'], params
        )

        return self._model.u.preds.val
