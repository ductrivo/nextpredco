import casadi as ca

from nextpredco.core.custom_types import SymVar


def create_f(
    x1: SymVar,
    x2: SymVar,
    u: SymVar,
    mu: SymVar,
) -> SymVar:
    x1_dot = x1 + u * (mu + (1 - mu) * x1)
    x2_dot = x2 + u * (mu + 4 * (1 - mu) * x2)
    return ca.vertcat(x1_dot, x2_dot)


# @abstractmethod
def create_g(
    x1: SymVar,
    x2: SymVar,
    u: SymVar,
    mu: SymVar,
) -> SymVar:
    # TODO
    return SymVar('g', 0)
