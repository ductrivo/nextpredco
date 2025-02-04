import casadi as ca

from nextpredco.core.custom_types import Symbolic


def create_f(
    c_a: Symbolic,
    c_b: Symbolic,
    t_r: Symbolic,
    t_k: Symbolic,
    f: Symbolic,
    q_dot: Symbolic,
    alpha: Symbolic,
    beta: Symbolic,
    k0_ab: Symbolic,
    k0_bc: Symbolic,
    k0_ad: Symbolic,
    rgas: Symbolic,
    ea_ab: Symbolic,
    ea_bc: Symbolic,
    ea_ad: Symbolic,
    hr_ab: Symbolic,
    hr_bc: Symbolic,
    hr_ad: Symbolic,
    rou: Symbolic,
    cp: Symbolic,
    cp_k: Symbolic,
    ar: Symbolic,
    vr: Symbolic,
    mk: Symbolic,
    tin: Symbolic,
    kw: Symbolic,
    c_a0: Symbolic,
) -> Symbolic:
    k1 = beta * k0_ab * ca.exp((-ea_ab) / (t_r + 273.15))
    k2 = k0_bc * ca.exp((-ea_bc) / (t_r + 273.15))
    k3 = k0_ad * ca.exp((-alpha * ea_ad) / (t_r + 273.15))
    t_diff = t_r - t_k

    ca_dot = f * (c_a0 - c_a) - k1 * c_a - k3 * (c_a**2)
    cb_dot = -f * c_b + k1 * c_a - k2 * c_b
    tr_dot = (
        (
            (k1 * c_a * hr_ab + k2 * c_b * hr_bc + k3 * (c_a**2) * hr_ad)
            / (-rou * cp)
        )
        + f * (tin - t_r)
        + ((kw * ar) * (-t_diff)) / (rou * cp * vr)
    )
    tk_dot = (q_dot + kw * ar * t_diff) / (mk * cp_k)
    return ca.vertcat(ca_dot, cb_dot, tr_dot, tk_dot)


# @abstractmethod
def create_g(
    c_a: Symbolic,
    c_b: Symbolic,
    t_r: Symbolic,
    t_k: Symbolic,
    f: Symbolic,
    q_dot: Symbolic,
    alpha: Symbolic,
    beta: Symbolic,
    k0_ab: Symbolic,
    k0_bc: Symbolic,
    k0_ad: Symbolic,
    rgas: Symbolic,
    ea_ab: Symbolic,
    ea_bc: Symbolic,
    ea_ad: Symbolic,
    hr_ab: Symbolic,
    hr_bc: Symbolic,
    hr_ad: Symbolic,
    rou: Symbolic,
    cp: Symbolic,
    cp_k: Symbolic,
    ar: Symbolic,
    vr: Symbolic,
    mk: Symbolic,
    tin: Symbolic,
    kw: Symbolic,
    c_a0: Symbolic,
) -> Symbolic:
    # TODO
    return Symbolic('g', 0)
