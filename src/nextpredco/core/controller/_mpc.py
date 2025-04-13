from collections import OrderedDict
from copy import copy
from dataclasses import dataclass, field
from functools import reduce as reduce
from typing import override

import casadi as ca
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from rich.pretty import pretty_repr as pretty_repr

from nextpredco.core._consts import COST_ELEMENTS
from nextpredco.core._logger import logger
from nextpredco.core._sync import GlobalState
from nextpredco.core._typing import (
    Array2D,
    ArrayType,
    FloatType,
    IntType,
    PredDType,
    Symbolic,
)
from nextpredco.core.controller._controller import ControllerABC
from nextpredco.core.model._descriptors import (
    DataVariable,
    ReadOnly,
    VariableView,
    VariableViewDict2,
)
from nextpredco.core.model._model import ModelABC
from nextpredco.core.settings import (
    IntegratorSettings,
    MPCSettings,
    OptimizerSettings,
)


def get_y(x: ArrayType, y_vars: list[str]) -> ArrayType:
    y: ArrayLike = []

    indexes = GlobalState.get_indexes(vars_=y_vars)

    if isinstance(x, Symbolic):
        return x[indexes, :]
    return x[indexes, :]


def get_du(u_arr: ArrayType, u_last: ArrayType) -> ArrayType:
    du: ArrayLike = [u_arr[:, 0] - u_last]
    for i in range(u_arr.shape[1] - 1):
        du.append(u_arr[:, i + 1] - u_arr[:, i])

    if isinstance(u_arr, Symbolic):
        return ca.hcat(du)

    return np.array([du])


def get_full_data(instance, attr_name: str, k: IntType, n_rows: IntType):
    if k not in getattr(instance, attr_name):
        getattr(instance, attr_name)[k] = np.zeros((n_rows, instance.n_cols))
    return getattr(instance, attr_name)


class MPCViewDict:
    def __set_name__(self, owner, name: str) -> None:
        self._name = name

    def __get__(self, instance, owner):
        k = GlobalState.k()
        private_name = '_' + self._name

        if instance._prefix == 'cost_':
            full_data = get_full_data(
                instance=instance,
                attr_name=private_name,
                k=k,
                n_rows=1,
            )
            return VariableViewDict2(
                k=k,
                full_data=full_data,
                indexes=[0],
                columns=[f'{instance._prefix}{self._name}{instance._suffix}'],
            )

        if self._name in ['y']:
            columns = [
                f'{instance._prefix}{var}{instance._suffix}'
                for var in instance.y_vars
            ]

            if k not in instance._x:
                instance._x[k] = np.zeros(
                    (GlobalState.n('x'), instance.n_cols)
                )

            return VariableViewDict2(
                k=k,
                full_data=instance._x,
                indexes=GlobalState.get_indexes(instance.y_vars),
                columns=columns,
            )

        if self._name in ['k', 't']:
            full_data = get_full_data(
                instance=instance,
                attr_name=private_name,
                k=k,
                n_rows=1,
            )

            return VariableViewDict2(
                k=k,
                full_data=full_data,
                indexes=[0],
                columns=[f'{instance._prefix}{self._name}{instance._suffix}'],
            )

        if self._name in ['x', 'z', 'u', 'p', 'q', 'm', 'upq']:
            vars_ = getattr(GlobalState, f'{self._name}_vars')()
            indexes = getattr(GlobalState, f'{self._name}_indexes')()
            columns = [
                f'{instance._prefix}{var}{instance._suffix}' for var in vars_
            ]
            full_data = get_full_data(
                instance=instance,
                attr_name=private_name,
                k=k,
                n_rows=len(columns),
            )

        elif self._name in ['x_fine', 'z_fine']:
            vars_ = getattr(GlobalState, f'{self._name[0]}_vars')()
            indexes = GlobalState.get_indexes(vars_)
            columns = [
                f'{instance._prefix}{var}{instance._suffix}_fine'
                for var in vars_
            ]
            full_data = get_full_data(
                instance=instance,
                attr_name=private_name,
                k=k,
                n_rows=len(columns),
            )

        return VariableViewDict2(
            k=k,
            full_data=full_data,
            indexes=indexes,
            columns=columns,
        )


@dataclass
class GoalsData:
    y_vars: list[str]
    n_cols: IntType = 1

    _prefix = ''
    _suffix = '_goal'

    _u: PredDType = field(default_factory=OrderedDict)
    _x: PredDType = field(default_factory=OrderedDict)

    u = MPCViewDict()
    x = MPCViewDict()
    y = MPCViewDict()

    def get_var(self, var: str, full_data: PredDType) -> VariableViewDict2:
        if var in GlobalState.x_vars():
            return VariableViewDict2(
                k=GlobalState.k(),
                full_data=self._x,
                indexes=[GlobalState.x_vars().index(var)],
                columns=[var],
            )

        if var in GlobalState.u_vars():
            return VariableViewDict2(
                k=GlobalState.k(),
                full_data=self._u,
                indexes=[GlobalState.x_vars().index(var)],
                columns=[var],
            )

        msg = f"Variable '{var}' not found in x_vars, or u_vars."
        raise ValueError(msg)

    @property
    def df(self) -> pd.DataFrame:
        dfs = [self.x.df, self.u.df]

        return reduce(
            lambda left, right: pd.merge(left, right, on='k'),
            dfs,
        )


@dataclass
class PredictionsData:
    y_vars: list[str]
    n_cols: IntType

    _prefix = ''
    _suffix = '_pred'

    _k: PredDType = field(default_factory=OrderedDict)
    _t: PredDType = field(default_factory=OrderedDict)
    _x: PredDType = field(default_factory=OrderedDict)
    _z: PredDType = field(default_factory=OrderedDict)
    _u: PredDType = field(default_factory=OrderedDict)
    _x_fine: PredDType = field(default_factory=OrderedDict)
    _z_fine: PredDType = field(default_factory=OrderedDict)

    k = MPCViewDict()
    t = MPCViewDict()
    x = MPCViewDict()
    z = MPCViewDict()
    u = MPCViewDict()
    x_fine = MPCViewDict()
    z_fine = MPCViewDict()
    y = MPCViewDict()

    def get_var(self, var: str) -> VariableViewDict2:
        if var in GlobalState.x_vars():
            return VariableViewDict2(
                k=GlobalState.k(),
                full_data=self._x,
                indexes=[GlobalState.x_vars().index(var)],
                columns=[var],
            )

        if var in GlobalState.z_vars():
            return VariableViewDict2(
                k=GlobalState.k(),
                full_data=self._z,
                indexes=[GlobalState.z_vars().index(var)],
                columns=[var],
            )

        if var in GlobalState.u_vars():
            return VariableViewDict2(
                k=GlobalState.k(),
                full_data=self._u,
                indexes=[GlobalState.u_vars().index(var)],
                columns=[var],
            )

        msg = f"Variable '{var}' not found in x_vars, z_vars, or u_vars."
        raise ValueError(msg)

    @property
    def df(self) -> pd.DataFrame:
        dfs = [self.t.df, self.u.df, self.x.df, self.x_fine.df]

        return reduce(
            lambda left, right: pd.merge(left, right, on='k'),
            dfs,
        )


@dataclass
class CostsData:
    y_vars: list[str]
    n_cols: IntType = 1

    _prefix = 'cost_'
    _suffix = ''
    _x: PredDType = field(default_factory=OrderedDict)
    _y: PredDType = field(default_factory=OrderedDict)
    _u: PredDType = field(default_factory=OrderedDict)
    _du: PredDType = field(default_factory=OrderedDict)
    _total: PredDType = field(default_factory=OrderedDict)

    x = MPCViewDict()
    y = MPCViewDict()
    u = MPCViewDict()
    du = MPCViewDict()
    total = MPCViewDict()

    @property
    def df(self) -> pd.DataFrame:
        dfs = [
            self.x.df,
            self.y.df,
            self.u.df,
            self.du.df,
            self.total.df,
        ]
        return reduce(
            lambda left, right: pd.merge(left, right, on='k'),
            dfs,
        )


@dataclass
class UpperBoundsData:
    y_vars: list[str]
    n_cols: IntType

    _prefix = ''
    _suffix = '_ub'

    _x: PredDType = field(default_factory=OrderedDict)
    _z: PredDType = field(default_factory=OrderedDict)
    _u: PredDType = field(default_factory=OrderedDict)

    x = MPCViewDict()
    z = MPCViewDict()
    u = MPCViewDict()
    y = MPCViewDict()


@dataclass
class LowerBoundsData:
    y_vars: list[str]
    n_cols: IntType

    _prefix = ''
    _suffix = '_lb'
    _x: PredDType = field(default_factory=OrderedDict)
    _z: PredDType = field(default_factory=OrderedDict)
    _u: PredDType = field(default_factory=OrderedDict)

    x = MPCViewDict()
    z = MPCViewDict()
    u = MPCViewDict()
    y = MPCViewDict()


class BoundsData:
    def __init__(self, y_vars: list[str], n_cols: IntType):
        self._upper = UpperBoundsData(y_vars=y_vars, n_cols=n_cols)
        self._lower = LowerBoundsData(y_vars=y_vars, n_cols=n_cols)

    @property
    def upper(self) -> UpperBoundsData:
        return self._upper

    @property
    def lower(self) -> LowerBoundsData:
        return self._lower


@dataclass
class MPCData:
    goals: GoalsData
    bounds: BoundsData
    predictions: PredictionsData
    costs: CostsData

    @classmethod
    def create(cls, y_vars: list[str], n_cols: IntType):
        """Factory method to create MPCData with consistent y_vars for all components."""
        return cls(
            goals=GoalsData(y_vars=y_vars),
            bounds=BoundsData(y_vars=y_vars, n_cols=n_cols),
            predictions=PredictionsData(y_vars=y_vars, n_cols=n_cols),
            costs=CostsData(y_vars=y_vars),
        )


class MPC(ControllerABC):
    n_pred = ReadOnly[IntType]()

    settings = ReadOnly[MPCSettings]()

    data = ReadOnly[MPCData]()
    goals = ReadOnly[GoalsData]('data')
    bounds = ReadOnly[BoundsData]('data')
    predictions = ReadOnly[PredictionsData]('data')
    costs = ReadOnly[CostsData]('data')

    def __init__(
        self,
        settings: MPCSettings,
        model: ModelABC,
        optimizer_settings: OptimizerSettings | None = None,
        integrator_settings: IntegratorSettings | None = None,
    ):
        super().__init__(
            settings,
            model,
            optimizer_settings,
            integrator_settings,
        )
        # To ensure that auto completion only shows MPCSettings attributes
        self._settings: MPCSettings = settings
        self._n_pred = settings.n_pred
        self._y_vars = GlobalState.y_vars()

        self._weights: dict[str, FloatType | Array2D] = {
            'x': settings.weight_x[0],
            'y': settings.weight_y[0],
            'u': settings.weight_u[0],
            'du': np.diag(settings.weight_du),
        }

        self._data = MPCData.create(
            y_vars=self._y_vars,
            n_cols=self._settings.n_pred,
        )

        self._inputs = {
            'x': [self._model.x, 'vec'],
            'u': [self._model.u, 'vec'],
            'p': [self._model.p, 'vec'],
            'q': [self._model.q, 'vec'],
            'y_goal': [self.goals.y, 'arr'],
        }
        self._outputs = {'u': [self.predictions.u, 'vec']}
        self._scale = {
            'x': 1,
            'u': 1,
        }
        # TODO Verify when need to re-create the nlp solver
        self._nlp_solver: ca.Function
        self._nlp_funcs: dict[str, ca.Function]

        logger.debug('Creating nlp solver...')
        self._nlp_solver, self._nlp_funcs = self._create_nlp_solver()
        logger.debug('Created nlp solver')

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
            'y': get_y(x=x_preds, y_vars=self._y_vars),
            'du': get_du(u_arr=u_preds, u_last=u_last),
        }
        goals = {
            'x': x_goal,
            'u': u_goal,
            'y': get_y(x_goal, y_vars=self._y_vars),
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
        x0 = Symbolic.sym('__x0', GlobalState.n('x'), 1)
        z0 = Symbolic.sym('__z0', GlobalState.n('z'), 1)
        u0 = Symbolic.sym('__u0', GlobalState.n('u'), 1)

        p0 = Symbolic.sym('__p0', GlobalState.n('p'), 1)
        q0 = Symbolic.sym('__q0', GlobalState.n('q'), 1)

        x_goal = Symbolic.sym('__x_goal', GlobalState.n('x'), 1)
        u_goal = Symbolic.sym('__u_goal', GlobalState.n('u'), 1)

        x_ub = Symbolic.sym('__x_ub', GlobalState.n('x'), 1)
        x_lb = Symbolic.sym('__x_lb', GlobalState.n('x'), 1)

        u_lb = Symbolic.sym('__u_lb', GlobalState.n('u'), 1)
        u_ub = Symbolic.sym('__u_ub', GlobalState.n('u'), 1)

        scale_u_arr = np.vstack([self._scale['u']] * self._n_pred)

        u_pred_vec_scaled: Symbolic = Symbolic.sym(
            'u_preds', GlobalState.n('u') * self._n_pred, 1
        )

        u_pred_vec = u_pred_vec_scaled * scale_u_arr
        logger.debug(f'u_pred_vec_scaled = {u_pred_vec_scaled.shape}')

        logger.debug('Reshaping predicted controls')
        u_preds = u_pred_vec.reshape((GlobalState.n('u'), self._n_pred))

        upq_arr_ = []
        for k in range(self._n_pred):
            upq_ = self._model.get_upq(
                u_arr=u_preds[:, k],
                p_arr=p0,
                q_arr=q0,
            )
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
        costs = self.compute_costs(
            x_preds / self._scale['x'],
            u_preds / self._scale['u'],
            x_goal / self._scale['x'],
            u_goal / self._scale['u'],
            u0 / self._scale['u'],
        )

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
            'x': u_pred_vec_scaled,
            'f': costs['total'],
            'g': ca.vcat(constraints),
            'p': params,
        }

        logger.debug('Creating NLP functions')
        nlp_outputs = {
            'x_preds': x_preds,
            'z_preds': Symbolic.sym(
                '__z_preds', GlobalState.n('z'), self._n_pred
            ),
            'u_lb': ca.vcat([u_lb] * self._n_pred),
            'u_ub': ca.vcat([u_ub] * self._n_pred),
        }

        print(f'nlp_outputs = {nlp_outputs}')

        nlp_funcs = {
            key: ca.Function(
                f'{key}_func',
                [u_pred_vec_scaled, params],
                [output],
            )
            for key, output in nlp_outputs.items()
        }

        for key, val in costs.items():
            nlp_funcs[f'cost_{key}'] = ca.Function(
                f'{key}_func',
                [u_pred_vec_scaled, params],
                [val],
            )

        logger.debug('Creating NLP solver')
        # input(f'nlp = {pretty_repr(nlp)}')
        nlp_solver: ca.Function = ca.nlpsol(
            'nlp_solver',
            'ipopt',
            nlp,
            self._optimizer.settings.opts,
        )
        return nlp_solver, nlp_funcs

    def make_step(self):
        u_guess = np.vstack([self._model.u.vec] * self._n_pred)
        x0 = self._model.x.vec
        z0 = self._model.z.vec
        u0 = self._model.u.vec
        p0 = self._model.p.vec
        q0 = self._model.q.vec
        x_goal = self.goals.x.arr
        u_goal = self.goals.u.arr
        x_lb = self.bounds.lower.x.vec
        x_ub = self.bounds.upper.x.vec
        u_lb = self.bounds.lower.u.vec
        u_ub = self.bounds.upper.u.vec

        # x_lb = np.array([[0.1, 0.1, 50, 50]]).T
        # x_ub = np.array([[2, 2, 140, 140]]).T
        # u_lb = np.array([[5, -8500]]).T
        # u_ub = np.array([[100, 0.0]]).T

        params = np.vstack(
            [x0, z0, u0, p0, q0, x_goal, u_goal, x_lb, x_ub, u_lb, u_ub]
        )
        # logger.debug(pretty_repr(params.T))
        # input('Pred')
        # logger.debug(
        #     pretty_repr(
        #         dict(
        #             u_guess=u_guess,
        #             scale_u=self._scale['u'],
        #             x0=u_guess / self._scale['u'],
        #             p=params,
        #             lbx=self._nlp_funcs['u_lb'](u_guess, params),
        #             ubx=self._nlp_funcs['u_ub'](u_guess, params),
        #         )
        #     )
        # )
        sol_ = self._nlp_solver(
            x0=u_guess / np.vstack([self._scale['u']] * self._n_pred),
            p=params,
            lbx=np.vstack([u_lb] * self._n_pred)
            / np.vstack([self._scale['u']] * self._n_pred),
            ubx=np.vstack([u_ub] * self._n_pred)
            / np.vstack([self._scale['u']] * self._n_pred),
            ubg=0,
            # lbg=0,
        )

        # logger.debug(pretty_repr(sol_))
        # input('sol')
        sol: dict[str, NDArray]

        sol = {key: val.full() for key, val in sol_.items()}

        vals = {
            key: func(sol['x'], params).full()
            for key, func in self._nlp_funcs.items()
        }

        k_arr = GlobalState.k() + 1 + np.arange(self._n_pred).reshape((1, -1))
        t_arr = k_arr * self._settings.dt
        u_arr = (
            sol['x'].reshape((-1, self._n_pred), order='F') * self._scale['u']
        )

        # input(f'u_arr = {u_arr}')
        # Verify
        upq_arr_ = []
        for k in range(self._n_pred):
            upq_ = self._model.get_upq(
                u_arr=u_arr[:, k, None],
                p_arr=p0,
                q_arr=q0,
            )
            upq_arr_.append(upq_)
        upq_arr = ca.hcat(upq_arr_)

        # logger.debug(f'x0 _ taylor:\n{x0}')
        _, _, x_preds, _ = self._integrator.integrate(
            x0=x0,
            z0=z0,
            upq_arr=upq_arr,
            t_grid=np.arange(self._n_pred + 1) * self._settings.dt,
        )

        # logger.debug(f'x0 _ idas:\n{x0}')
        _, _, x_fine_arr, z_fine_arr = self._model.compute_xz(
            x0=x0,
            z0=z0,
            u_arr=u_arr,
            p_arr=p0,
            q_arr=q0,
            t_grid=np.arange(self._n_pred + 1) * self._settings.dt,
        )
        # logger.debug(f'x_preds =\n{x_preds}\nx_fine_arr =\n{x_fine_arr}')
        # input('compare')
        # logger.debug(
        #     pretty_repr(
        #         dict(
        #             x0=x0.T,
        #             z0=z0,
        #             u_arr=u_arr,
        #             p_arr=p0,
        #             q_arr=q0,
        #             t_grid=[
        #                 *self._model.t.vec[0, :].tolist(),
        #                 *t_arr[0, :].tolist(),
        #             ],
        #             x_fine_arr=x_fine_arr,
        #         )
        #     )
        # )
        # input('Pred')
        self.predictions.k.arr = k_arr
        self.predictions.k.arr = k_arr
        self.predictions.t.arr = t_arr
        self.predictions.u.arr = u_arr
        self.predictions.x.arr = vals['x_preds']
        self.predictions.z.arr = vals['z_preds']
        self.predictions.x_fine.arr = x_fine_arr
        self.predictions.z_fine.arr = z_fine_arr

        for key in [*COST_ELEMENTS]:
            attr = getattr(self.costs, key)
            attr.arr = vals[f'cost_{key}']

        return self.predictions.u.vec
