import casadi as ca

from nextpredco.core._typing import Symbolic


def create_f(
    x1: Symbolic,
    x2: Symbolic,
    u: Symbolic,
    mu: Symbolic,
) -> Symbolic:
    x1_dot = x1 + u * (mu + (1 - mu) * x1)
    x2_dot = x2 + u * (mu + 4 * (1 - mu) * x2)
    return ca.vertcat(x1_dot, x2_dot)


# @abstractmethod
def create_g(
    x1: Symbolic,
    x2: Symbolic,
    u: Symbolic,
    mu: Symbolic,
) -> Symbolic:
    # TODO
    return Symbolic('g', 0)
