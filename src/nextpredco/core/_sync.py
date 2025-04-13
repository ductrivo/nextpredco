from threading import Lock, Thread, current_thread
from typing import TYPE_CHECKING, Literal

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from nextpredco.core._typing import (
    Array2D,
    ArrayType,
    FloatType,
    IntType,
    PredDType,
    SourceType,
    Symbolic,
    TgridType,
)

if TYPE_CHECKING:
    from nextpredco.core.settings import ModelSettings


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    https://refactoring.guru/design-patterns/singleton/python/example#example-1
    """

    _instances = {}  # type: ignore[var-annotated] # noqa

    __lock: Lock = Lock()
    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.

        with cls.__lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field
            # is already initialized, the thread won't create a new object.
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


PLANT_VARS = ['x', 'z', 'u', 'p', 'q', 'm', 'upq', 'y']


class GlobalState(metaclass=SingletonMeta):
    __lock = Lock()
    _k: IntType
    _k_max: IntType

    _dt: FloatType
    _t_max: FloatType

    _x_vars: list[str]
    _z_vars: list[str]
    _u_vars: list[str]
    _p_vars: list[str]
    _q_vars: list[str]
    _m_vars: list[str]
    _upq_vars: list[str]
    _const_vars: list[str]
    _y_vars: list[str]

    _x_indexes: list[int]
    _z_indexes: list[int]
    _u_indexes: list[int]
    _p_indexes: list[int]
    _q_indexes: list[int]
    _m_indexes: list[int]
    _upq_indexes: list[int]
    _y_indexes: list[int]

    @classmethod
    def create(
        cls,
        plant_settings: 'ModelSettings',
        # k: IntType,
        # dt: FloatType,
        # t_max: FloatType,
        # x_vars: list[str],
        # z_vars: list[str],
        # u_vars: list[str],
        # p_vars: list[str],
        # q_vars: list[str],
        # m_vars: list[str],
        # consts: list[str],
    ):
        """Initialize the singleton instance variables."""
        with cls.__lock:
            vars_ = [f'{key}_vars' for key in PLANT_VARS]

            for var in ['k', 'dt', 't_max', *vars_]:
                setattr(cls, '_' + var, getattr(plant_settings, var))

            cls._const_vars = list(plant_settings.info.keys())

            cls._k_max = round(cls._t_max / cls._dt)

            # Initialize index mappings
            cls._update_indexes()

    @classmethod
    def k_max(cls):
        return cls._k_max

    @classmethod
    def k(cls) -> IntType:
        return cls._k

    @classmethod
    def dt(cls) -> FloatType:
        return cls._dt

    @classmethod
    def set_dt(cls, val: FloatType):
        with cls.__lock:
            cls._dt = val

    @classmethod
    def t_max(cls) -> FloatType:
        return cls._t_max

    @classmethod
    def set_k(cls, val: IntType):
        with cls.__lock:
            cls._k = val

    @classmethod
    def update_k(cls):
        with cls.__lock:
            cls._k += 1

    @classmethod
    def get_vars(
        cls,
        name: str | Literal['x', 'z', 'u', 'p', 'q', 'm', 'upq', 'y'],
    ) -> list[str] | None:
        """Retrieve a class attribute dynamically."""
        return getattr(cls, f'_{name}_vars', None)

    @classmethod
    def set_vars(
        cls,
        name: str | Literal['x', 'z', 'u', 'p', 'q', 'm', 'upq', 'y'],
        val: list[str],
    ):
        """Set a class attribute dynamically."""
        with cls.__lock:
            setattr(cls, f'_{name}_vars', val)

    @classmethod
    def x_vars(cls) -> list[str]:
        return cls._x_vars

    @classmethod
    def z_vars(cls) -> list[str]:
        return cls._z_vars

    @classmethod
    def u_vars(cls) -> list[str]:
        return cls._u_vars

    @classmethod
    def p_vars(cls) -> list[str]:
        return cls._p_vars

    @classmethod
    def q_vars(cls) -> list[str]:
        return cls._q_vars

    @classmethod
    def upq_vars(cls) -> list[str]:
        return cls._upq_vars

    @classmethod
    def m_vars(cls) -> list[str]:
        return cls._m_vars

    @classmethod
    def const_vars(cls) -> list[str]:
        return cls._const_vars

    @classmethod
    def y_vars(cls) -> list[str]:
        return cls._y_vars

    @classmethod
    def x_indexes(cls) -> list[int]:
        return cls._x_indexes

    @classmethod
    def z_indexes(cls) -> list[int]:
        return cls._z_indexes

    @classmethod
    def u_indexes(cls) -> list[int]:
        return cls._u_indexes

    @classmethod
    def p_indexes(cls) -> list[int]:
        return cls._p_indexes

    @classmethod
    def q_indexes(cls) -> list[int]:
        return cls._q_indexes

    @classmethod
    def m_indexes(cls) -> list[int]:
        return cls._m_indexes

    @classmethod
    def upq_indexes(cls) -> list[int]:
        return cls._upq_indexes

    @classmethod
    def y_indexes(cls) -> list[int]:
        return cls._y_indexes

    @classmethod
    def n(cls, name: Literal['x', 'z', 'u', 'p', 'q', 'm', 'upq', 'y']) -> int:
        return len(getattr(cls, f'_{name}_vars'))

    @classmethod
    def _update_indexes(cls):
        for ss_var in ['x', 'z', 'm', 'u', 'p', 'q', 'upq', 'y']:
            vars_ = getattr(cls, f'_{ss_var}_vars')
            indexes = cls.get_indexes(vars_)
            setattr(cls, f'_{ss_var}_indexes', indexes)

    @classmethod
    def get_indexes(cls, vars_: list[str]) -> list[int]:
        if set(vars_).issubset(set(cls._x_vars)):
            return [cls._x_vars.index(var) for var in vars_]
        if set(vars_).issubset(set(cls._z_vars)):
            return [cls._z_vars.index(var) for var in vars_]
        if set(vars_).issubset(set(cls._upq_vars)):
            return [cls._upq_vars.index(var) for var in vars_]

        msg = 'Variables not found in x, z, or upq.'
        raise ValueError(msg)


def worker():
    singleton = GlobalState()
    print(f'Thread {current_thread().name}: k = {singleton.get_k()}')


# For Unit test
if __name__ == '__main__':
    singleton = GlobalState()
    singleton.set_k(42)

    threads = [Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
