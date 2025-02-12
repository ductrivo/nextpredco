from copy import copy
from typing import override

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray
from rich.pretty import pretty_repr as pretty_repr

from nextpredco.core._consts import COST_ELEMENTS
from nextpredco.core._logger import logger
from nextpredco.core._typing import (
    Array2D,
    ArrayType,
    FloatType,
    IntType,
    Symbolic,
)
from nextpredco.core.controller import ControllerABC
from nextpredco.core.model import Model
from nextpredco.core.model._descriptors import ReadOnly2
from nextpredco.core.settings import (
    IntegratorSettings,
    MPCSettings,
    OptimizerSettings,
)


class MPC(ControllerABC):
    n_pred = ReadOnly2[IntType]()

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
        self._weights: dict[str, FloatType | Array2D] = {
            'x': settings.weight_x[0],
            'y': settings.weight_y[0],
            'u': settings.weight_u[0],
            'du': np.diag(settings.weight_du),
        }

        self._create_data_preds_in_model()

        # TODO Verify when need to re-create the nlp solver
        self._nlp_solver: ca.Function
        self._nlp_funcs: dict[str, ca.Function]

        logger.debug('Creating nlp solver...')
        self._nlp_solver, self._nlp_funcs = self._create_nlp_solver()
        logger.debug('Created nlp solver')

    def _create_data_preds_in_model(self):
        k_stamps = np.arange(
            start=0,
            stop=self._model.k_max + self._n_pred,
            step=self._n_pred,
            dtype=int,
        )

        # k_preds = {k: np.zeros((1, self.n_pred)) for k in k_stamps}

        # t_preds = {k: np.zeros((1, self.n_pred)) for k in k_stamps}

        # u_preds = {
        #     k: np.zeros((self.model.n('u'), self.n_pred)) for k in k_stamps
        # }
        # x_preds = {
        #     k: np.zeros((self.model.n('x'), self.n_pred)) for k in k_stamps
        # }

        # costs_full = {
        #     k: np.zeros((len(COST_ELEMENTS), self.n_pred)) for k in k_stamps
        # }

        # self._model.create_data_preds_full(
        #     k_preds_full=k_preds,
        #     t_preds_full=t_preds,
        #     u_preds_full=u_preds,
        #     x_preds_full=x_preds,
        #     costs_full=costs_full,
        # )

        # self._model.create_data_preds_full(
        #     k_preds_full={},
        #     t_preds_full={},
        #     u_preds_full={},
        #     x_preds_full={},
        #     costs_full={},
        # )

    def _create_nlp_solver(self):
        return self._single_shooting()

    def compute_costs(
        self,
        x_preds: ArrayType,
        u_preds: ArrayType,
        x_goal: ArrayLike,
        u_goal: ArrayLike,
        u_last: ArrayLike,
    ) -> dict[str, ArrayType]:
        arrs = {
            'x': x_preds,
            'u': u_preds,
            'y': self._model.get_y(x=x_preds),
            'du': self._model.get_du(u_arr=u_preds, u_last=u_last),
        }
        goals = {
            'x': x_goal,
            'u': u_goal,
            'y': self.model.get_y(x_goal),
            'du': None,
        }
        costs = {
            key: self._compute_cost_ingredients(
                arr=arrs[key],
                weight=self._weights[key],
                goal=goals[key],
            )
            for key in arrs
        }

        total = ca.sum1(ca.vcat(costs.values()))
        costs['total'] = total

        # logger.debug('x_preds: %s', x_preds)
        # logger.debug('costs: %s', costs)
        # input('Press Enter to continue')
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
        # if self._settings.normalizing:
        #     if isinstance(goal, Symbolic):
        #         logger.warning(
        #             'Goal of type %s. Normalizing is disabled.',
        #             Symbolic,
        #         )
        #     elif isinstance(goal, np.ndarray) and any(np.isclose(goal, 0)):
        #         logger.warning(
        #             'Goal has zero element(s). Normalizing is disabled.'
        #         )
        #         return arr - goal

        #     return arr / goal - 1
        return arr - goal

    def _single_shooting(self) -> ca.Function:
        logger.debug('Initializing symbols for single shooting')
        x0 = Symbolic.sym('__x0', self._model.n('x'), 1)
        z0 = Symbolic.sym('__z0', 0)
        u0 = Symbolic.sym('__u0', self._model.n('u'), 1)

        p0 = Symbolic.sym('__p0', self._model.n('p'), 1)
        q0 = Symbolic.sym('__q0', self._model.n('q'), 1)

        x_goal = Symbolic.sym('__x_goal', self._model.n('x'), 1)
        u_goal = Symbolic.sym('__u_goal', self._model.n('u'), 1)

        x_ub = Symbolic.sym('__x_ub', self._model.n('x'), 1)
        x_lb = Symbolic.sym('__x_lb', self._model.n('x'), 1)

        u_lb = Symbolic.sym('__u_lb', self._model.n('u'), 1)
        u_ub = Symbolic.sym('__u_ub', self._model.n('u'), 1)

        u_pred_vec: Symbolic = Symbolic.sym(
            'u_preds', self._model.n('u') * self._n_pred, 1
        )

        logger.debug('Reshaping predicted controls')
        u_preds = u_pred_vec.reshape((self._model.n('u'), self._n_pred))
        upq_arr_ = []
        for k in range(self._n_pred):
            upq_ = self._model.get_upq(u_arr=u_preds[:, k], p_arr=p0, q_arr=q0)
            upq_arr_.append(upq_)
        upq_arr = ca.hcat(upq_arr_)

        logger.debug('Integrating with initial states')
        _, _, x_preds, _ = self._integrator.integrate(
            x0=x0,
            z0=z0,
            upq_arr=upq_arr,
            t_grid=np.arange(self._n_pred + 1) * self._settings.dt,
        )

        logger.debug('Computing costs')
        costs = self.compute_costs(x_preds, u_preds, x_goal, u_goal, u0)

        logger.debug('Setting up constraints')
        constraints: list[Symbolic] = []
        for k in range(x_preds.shape[1]):
            constraints.append(x_lb - x_preds[:, k])
            constraints.append(x_preds[:, k] - x_ub)

        logger.debug('Creating NLP parameters')
        params = ca.vcat(
            [x0, z0, u0, p0, q0, x_goal, u_goal, x_lb, x_ub, u_lb, u_ub]
        )
        nlp = {
            'x': u_pred_vec,
            'f': costs['total'],
            'g': ca.vcat(constraints),
            'p': params,
        }

        logger.debug('Creating NLP functions')
        nlp_outputs = {
            'x_preds': x_preds,
            'z_preds': Symbolic.sym(
                '__z_preds', self._model.n('z'), self._n_pred
            ),
            'u_lb': ca.vcat([u_lb] * self._n_pred),
            'u_ub': ca.vcat([u_ub] * self._n_pred),
        }

        nlp_funcs = {
            key: ca.Function(
                f'{key}_func',
                [u_pred_vec, params],
                [output],
            )
            for key, output in nlp_outputs.items()
        }

        for key, val in costs.items():
            nlp_funcs[f'cost_{key}'] = ca.Function(
                f'{key}_func',
                [u_pred_vec, params],
                [val],
            )

        logger.debug('Creating NLP solver')
        # input(f'nlp = {pretty_repr(nlp)}')
        nlp_solver: ca.Function = ca.nlpsol('nlp_solver', 'ipopt', nlp)
        return nlp_solver, nlp_funcs

    def make_step(self):
        u_guess = np.repeat(self._model.u.est.val, self._n_pred)
        x0 = self._model.x.est.val
        z0 = self._model.z.est.val
        u0 = self._model.u.est.val
        p0 = self._model.p.est.val
        q0 = self._model.q.est.val
        x_goal = self._model.x.goal.val
        u_goal = self._model.u.goal.val
        x_lb = np.array([[0.1, 0.1, 50, 50]]).T
        x_ub = np.array([[2, 2, 140, 140]]).T
        u_lb = np.array([[5, -8500]]).T
        u_ub = np.array([[100, 0.0]]).T

        params = np.vstack(
            [x0, z0, u0, p0, q0, x_goal, u_goal, x_lb, x_ub, u_lb, u_ub]
        )
        # logger.debug(
        #     pretty_repr(
        #         dict(
        #             u_guess=np.repeat(self._model.u.est.val, self._n_pred),
        #             x0=self._model.x.est.val,
        #             z0=self._model.z.est.val,
        #             u0=self._model.u.est.val,
        #             p0=self._model.p.est.val,
        #             q0=self._model.q.est.val,
        #             x_goal=self._model.x.goal.val,
        #             u_goal=self._model.u.goal.val,
        #             x_lb=np.array([[0.1, 0.1, 50, 50]]).T,
        #             x_ub=np.array([[2, 2, 140, 140]]).T,
        #             u_lb=np.array([[5, -8500]]).T,
        #             u_ub=np.array([[100, 0.0]]).T,
        #         )
        #     )
        # )
        # input('Press Enter to start solver')
        # input(self._nlp_funcs['u_lb'](u_guess, params))
        # input(self._nlp_funcs['u_ub'](u_guess, params))
        sol_ = self._nlp_solver(
            x0=u_guess,
            p=params,
            lbx=self._nlp_funcs['u_lb'](u_guess, params),
            ubx=self._nlp_funcs['u_ub'](u_guess, params),
            ubg=0,
        )

        sol: dict[str, NDArray]
        sol = {key: val.full() for key, val in sol_.items()}

        vals = {
            key: func(sol['x'], params).full()
            for key, func in self._nlp_funcs.items()
        }

        k_arr = self._model.k + 1 + np.arange(self._n_pred).reshape((1, -1))
        t_arr = k_arr * self._settings.dt
        u_arr = sol['x'].reshape((-1, self._n_pred), order='F')

        _, _, x_fine_arr, z_fine_arr = self._model.compute_xz(
            x0=x0,
            z0=z0,
            u_arr=u_arr,
            p_arr=p0,
            q_arr=q0,
            t_grid=[*self._model.t.val[0, :].tolist(), *t_arr[0, :].tolist()],
        )

        self._model.predictions.k.arr = k_arr
        self._model.predictions.t.arr = t_arr
        self._model.predictions.u.arr = u_arr
        self._model.predictions.x.arr = vals['x_preds']
        self._model.predictions.z.arr = vals['z_preds']
        self._model.predictions.x_fine.arr = x_fine_arr
        self._model.predictions.z_fine.arr = z_fine_arr

        for key in [*COST_ELEMENTS]:
            attr = getattr(self._model.predictions.costs, key)
            attr.arr = vals[f'cost_{key}']

        # input(f'vals = {vals}')
        return self._model.predictions.u.val
