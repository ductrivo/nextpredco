from collections.abc import Callable

import casadi as ca

from nextpredco.core.custom_types import SymVar


def create_f(
    cA: SymVar,
    cB: SymVar,
    tR: SymVar,
    tK: SymVar,
    f: SymVar,
    q_dot: SymVar,
    alpha: SymVar,
    beta: SymVar,
    k0_ab: SymVar,
    k0_bc: SymVar,
    k0_ad: SymVar,
    rGas: SymVar,
    eA_ab: SymVar,
    eA_bc: SymVar,
    eA_ad: SymVar,
    hR_ab: SymVar,
    hR_bc: SymVar,
    hR_ad: SymVar,
    rou: SymVar,
    cP: SymVar,
    cP_k: SymVar,
    aR: SymVar,
    vR: SymVar,
    mK: SymVar,
    tIn: SymVar,
    kW: SymVar,
    cA0: SymVar,
) -> SymVar:
    k1 = beta * k0_ab * ca.exp((-eA_ab) / (tR + 273.15))
    k2 = k0_bc * ca.exp((-eA_bc) / (tR + 273.15))
    k3 = k0_ad * ca.exp((-alpha * eA_ad) / (tR + 273.15))
    t_diff = tR - tK

    cA_dot = f * (cA0 - cA) - k1 * cA - k3 * (cA**2)
    cB_dot = -f * cB + k1 * cA - k2 * cB
    tR_dot = (
        (
            (k1 * cA * hR_ab + k2 * cB * hR_bc + k3 * (cA**2) * hR_ad)
            / (-rou * cP)
        )
        + f * (tIn - tR)
        + ((kW * aR) * (-t_diff)) / (rou * cP * vR)
    )
    tK_dot = (q_dot + kW * aR * (t_diff)) / (mK * cP_k)
    return ca.vertcat(cA_dot, cB_dot, tR_dot, tK_dot)


# @abstractmethod
def create_g(
    cA: SymVar,
    cB: SymVar,
    tR: SymVar,
    tK: SymVar,
    f: SymVar,
    q_dot: SymVar,
    alpha: SymVar,
    beta: SymVar,
    k0_ab: SymVar,
    k0_bc: SymVar,
    k0_ad: SymVar,
    rGas: SymVar,
    eA_ab: SymVar,
    eA_bc: SymVar,
    eA_ad: SymVar,
    hR_ab: SymVar,
    hR_bc: SymVar,
    hR_ad: SymVar,
    rou: SymVar,
    cP: SymVar,
    cP_k: SymVar,
    aR: SymVar,
    vR: SymVar,
    mK: SymVar,
    tIn: SymVar,
    kW: SymVar,
    cA0: SymVar,
) -> SymVar:
    # TODO
    return SymVar('g', 0)
