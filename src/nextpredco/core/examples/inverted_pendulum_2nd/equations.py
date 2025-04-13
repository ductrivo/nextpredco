import casadi as ca

from nextpredco.core._typing import Symbolic


def create_f(
    # x
    x: Symbolic,
    theta1: Symbolic,
    theta2: Symbolic,
    dx: Symbolic,
    dtheta1: Symbolic,
    dtheta2: Symbolic,
    # z
    ddx: Symbolic,
    ddtheta1: Symbolic,
    ddtheta2: Symbolic,
    # u
    f: Symbolic,
    # constants
    m0: Symbolic,
    m1: Symbolic,
    m2: Symbolic,
    length_rod1: Symbolic,
    length_rod2: Symbolic,
    g: Symbolic,
) -> Symbolic:
    # TODO: Should return a dictionary that corresponds to states
    return ca.vertcat(
        dx,
        dtheta1,
        dtheta2,
        ddx,
        ddtheta1,
        ddtheta2,
    )


# @abstractmethod
def create_g(
    # x
    x: Symbolic,
    theta1: Symbolic,
    theta2: Symbolic,
    dx: Symbolic,
    dtheta1: Symbolic,
    dtheta2: Symbolic,
    # z
    ddx: Symbolic,
    ddtheta1: Symbolic,
    ddtheta2: Symbolic,
    # u
    f: Symbolic,
    # constants
    m0: Symbolic,
    m1: Symbolic,
    m2: Symbolic,
    length_rod1: Symbolic,
    length_rod2: Symbolic,
    g: Symbolic,
) -> Symbolic:
    l1 = length_rod1 / 2
    l2 = length_rod2 / 2
    inertia1 = (m1 * l1**2) / 3
    inertia2 = (m2 * l2**2) / 3

    h1 = m0 + m1 + m2
    h2 = m1 * l1 + m2 * length_rod1
    h3 = m2 * l2
    h4 = m1 * l1**2 + m2 * length_rod1**2 + inertia1
    h5 = m2 * l2 * length_rod1
    h6 = m2 * l2**2 + inertia2
    h7 = (m1 * l1 + m2 * length_rod1) * g
    h8 = m2 * l2 * g

    # 1
    return ca.vertcat(
        # 1
        h1 * ddx
        + h2 * ddtheta1 * ca.cos(theta1)
        + h3 * ddtheta2 * ca.cos(theta2)
        - (
            h2 * dtheta1**2 * ca.sin(theta1)
            + h3 * dtheta2**2 * ca.sin(theta2)
            + f
        ),
        # 2
        h2 * ca.cos(theta1) * ddx
        + h4 * ddtheta1
        + h5 * ca.cos(theta1 - theta2) * ddtheta2
        - (h7 * ca.sin(theta1) - h5 * dtheta2**2 * ca.sin(theta1 - theta2)),
        # 3
        h3 * ca.cos(theta2) * ddx
        + h5 * ca.cos(theta1 - theta2) * ddtheta1
        + h6 * ddtheta2
        - (h5 * dtheta1**2 * ca.sin(theta1 - theta2) + h8 * ca.sin(theta2)),
    )
