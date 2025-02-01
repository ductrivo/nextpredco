import casadi as ca

from nextpredco.core.custom_types import SymVar


def create_f(
    c_a: SymVar,
    c_b: SymVar,
    t_r: SymVar,
    t_k: SymVar,
    f: SymVar,
    q_dot: SymVar,
    alpha: SymVar,
    beta: SymVar,
    k0_ab: SymVar,
    k0_bc: SymVar,
    k0_ad: SymVar,
    rgas: SymVar,
    ea_ab: SymVar,
    ea_bc: SymVar,
    ea_ad: SymVar,
    hr_ab: SymVar,
    hr_bc: SymVar,
    hr_ad: SymVar,
    rou: SymVar,
    cp: SymVar,
    cp_k: SymVar,
    ar: SymVar,
    vr: SymVar,
    mk: SymVar,
    tin: SymVar,
    kw: SymVar,
    ca0: SymVar,
) -> SymVar:
    k1 = beta * k0_ab * ca.exp((-ea_ab) / (t_r + 273.15))
    k2 = k0_bc * ca.exp((-ea_bc) / (t_r + 273.15))
    k3 = k0_ad * ca.exp((-alpha * ea_ad) / (t_r + 273.15))
    t_diff = t_r - t_k

    ca_dot = f * (ca0 - c_a) - k1 * c_a - k3 * (c_a**2)
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
    c_a: SymVar,
    c_b: SymVar,
    t_r: SymVar,
    t_k: SymVar,
    f: SymVar,
    q_dot: SymVar,
    alpha: SymVar,
    beta: SymVar,
    k0_ab: SymVar,
    k0_bc: SymVar,
    k0_ad: SymVar,
    rgas: SymVar,
    ea_ab: SymVar,
    ea_bc: SymVar,
    ea_ad: SymVar,
    hr_ab: SymVar,
    hr_bc: SymVar,
    hr_ad: SymVar,
    rou: SymVar,
    cp: SymVar,
    cp_k: SymVar,
    ar: SymVar,
    vr: SymVar,
    mk: SymVar,
    tin: SymVar,
    kw: SymVar,
    ca0: SymVar,
) -> SymVar:
    # TODO
    return SymVar('g', 0)
