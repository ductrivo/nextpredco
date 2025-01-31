# Add do_mpc to path. This is not necessary if it was installed via pip
import os
import sys

import casadi as ca

# Import do_mpc package:
import do_mpc  # type: ignore [import-not-found]
import matplotlib.pyplot as plt
import numpy as np

rel_do_mpc_path = os.path.join('..', '..', '..')  # noqa: PTH118
sys.path.append(rel_do_mpc_path)


# from nextpredco.core.logger import logger

# Define the model:
model_type = 'continuous'  # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

# States struct (optimization variables):
C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1, 1))
C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1, 1))
T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1, 1))
T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1, 1))

# Input struct (optimization variables):
F = model.set_variable(var_type='_u', var_name='F')
Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

# Certain parameters
K0_ab = 1.287e12  # K0 [h^-1]
K0_bc = 1.287e12  # K0 [h^-1]
K0_ad = 9.043e9  # K0 [l/mol.h]
R_gas = 8.3144621e-3  # Universal gas constant
E_A_ab = 9758.3 * 1.00  # * R_gas# [kj/mol]
E_A_bc = 9758.3 * 1.00  # * R_gas# [kj/mol]
E_A_ad = 8560.0 * 1.0  # * R_gas# [kj/mol]
H_R_ab = 4.2  # [kj/mol A]
H_R_bc = -11.0  # [kj/mol B] Exothermic
H_R_ad = -41.85  # [kj/mol A] Exothermic
Rou = 0.9342  # Density [kg/l]
Cp = 3.01  # Specific Heat capacity [kj/Kg.K]
Cp_k = 2.0  # Coolant heat capacity [kj/kg.k]
A_R = 0.215  # Area of reactor wall [m^2]
V_R = 10.01  # 0.01 # Volume of reactor [l]
m_k = 5.0  # Coolant mass[kg]
T_in = 130.0  # Temp of inflow [Celsius]
K_w = 4032.0  # [kj/h.m^2.K]
C_A0 = (
    (5.7 + 4.5) / 2.0 * 1.0
)  # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]

# Uncertain parameters:
alpha = model.set_variable(var_type='_p', var_name='alpha')
beta = model.set_variable(var_type='_p', var_name='beta')

# Auxiliary terms
K_1 = beta * K0_ab * ca.exp((-E_A_ab) / (T_R + 273.15))
K_2 = K0_bc * ca.exp((-E_A_bc) / (T_R + 273.15))
K_3 = K0_ad * ca.exp((-alpha * E_A_ad) / (T_R + 273.15))

# Additionally, we define an artificial variable of interest,
# that is not a state of the system, but will be later used for plotting:
T_dif = model.set_expression(expr_name='T_dif', expr=T_R - T_K)

model.set_rhs('C_a', F * (C_A0 - C_a) - K_1 * C_a - K_3 * (C_a**2))
model.set_rhs('C_b', -F * C_b + K_1 * C_a - K_2 * C_b)
model.set_rhs(
    'T_R',
    (
        (K_1 * C_a * H_R_ab + K_2 * C_b * H_R_bc + K_3 * (C_a**2) * H_R_ad)
        / (-Rou * Cp)
    )
    + F * (T_in - T_R)
    + (((K_w * A_R) * (-T_dif)) / (Rou * Cp * V_R)),
)
model.set_rhs('T_K', (Q_dot + K_w * A_R * (T_dif)) / (m_k * Cp_k))

# Build the model
model.setup()

# Define the controller:
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 5,
    'n_robust': 1,
    'open_loop': 0,
    't_step': 0.01,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 3,
    'collocation_ni': 2,
    'store_full_solution': True,
    'nlpsol_opts': {
        'print_time': False,
        'ipopt.print_level': 0,
        'ipopt.print_timing_statistics': 'no',
    },
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

# Set parameters for the MPC controller
mpc.set_param(**setup_mpc)
mpc.scaling['_x', 'T_R'] = 100
mpc.scaling['_x', 'T_K'] = 100
mpc.scaling['_u', 'Q_dot'] = 2000
mpc.scaling['_u', 'F'] = 100

# Define the cost function:
_x = model.x
mterm = (_x['C_b'] - 0.6) ** 2  # terminal cost
lterm = (_x['C_b'] - 0.6) ** 2  # stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)

mpc.set_rterm(F=0.1, Q_dot=1e-3)  # input penalty

# Constraints
# lower bounds of the states
mpc.bounds['lower', '_x', 'C_a'] = 0.1
mpc.bounds['lower', '_x', 'C_b'] = 0.1
mpc.bounds['lower', '_x', 'T_R'] = 50
mpc.bounds['lower', '_x', 'T_K'] = 50

# upper bounds of the states
mpc.bounds['upper', '_x', 'C_a'] = 2
mpc.bounds['upper', '_x', 'C_b'] = 2
mpc.bounds['upper', '_x', 'T_K'] = 140

# lower bounds of the inputs
mpc.bounds['lower', '_u', 'F'] = 5
mpc.bounds['lower', '_u', 'Q_dot'] = -8500

# upper bounds of the inputs
mpc.bounds['upper', '_u', 'F'] = 100
mpc.bounds['upper', '_u', 'Q_dot'] = 0.0

mpc.set_nl_cons(
    'T_R', _x['T_R'], ub=140, soft_constraint=True, penalty_term_cons=1e2
)

# Uncertain values
alpha_var = np.array([1.0, 1.05, 0.95])
beta_var = np.array([1.0, 1.1, 0.9])

mpc.set_uncertainty_values(alpha=alpha_var, beta=beta_var)
mpc.setup()

# Estimator
estimator = do_mpc.estimator.StateFeedback(model)
simulator = do_mpc.simulator.Simulator(model)
params_simulator = {
    'integration_tool': 'idas',
    'abstol': 1e-3,
    'reltol': 1e-3,
    't_step': 0.01,
}

simulator.set_param(**params_simulator)

# Undertain parameters
p_num = simulator.get_p_template()
tvp_num = simulator.get_tvp_template()


# function for time-varying parameters
def tvp_fun(t_now):
    return tvp_num


# uncertain parameters
p_num['alpha'] = 1
p_num['beta'] = 1


def p_fun(t_now):
    return p_num


simulator.set_tvp_fun(tvp_fun)
simulator.set_p_fun(p_fun)

simulator.setup()

# Set the initial state of mpc, simulator and estimator:
C_a_0 = 0.8  # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5  # This is the controlled variable [mol/l]
T_R_0 = 134.14  # [C]
T_K_0 = 130.0  # [C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1, 1)

mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

for k in range(3):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
    print(f'k={k}')
    print(f'x = {mpc.data["_x"]}')
    print(f'u = {mpc.data["_u"]}')

# print(mpc.data['_time'])
# fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
# ax[0].plot(mpc.data['_time'], mpc.data['_x', 'C_a'])
# ax[0].plot(mpc.data['_time'], mpc.data['_x', 'C_b'])


# data_out = {'x': mpc.data['_x'], 'u': mpc.data['_u'], 'p': mpc.data['_p']}
# np.savez('transient_data.npz', **data_out)

# plt.show()
