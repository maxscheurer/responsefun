from sympy import symbols
from sympy.physics.quantum.operator import Operator
from responsefun.ResponseOperator import (
    MTM, S2S_MTM, ResponseVector, OneParticleOperator, TransitionFrequency
)

# ground state and excited state f
O, f = symbols("O, f", real=True)

# damping factor
gamma = symbols("gamma", real=True)

# indices of summation
n, m, p, k = symbols("n, m, p, k", real=True)

# external frequencies
w, w_o, w_1, w_2, w_3 = symbols("w, w_sigma, w_1, w_2, w_3", real=True)

# transition frequencies
w_f = TransitionFrequency(f, real=True)
w_n = TransitionFrequency(n, real=True)
w_m = TransitionFrequency(m, real=True)
w_p = TransitionFrequency(p, real=True)
w_k = TransitionFrequency(k, real=True)

# electric dipole operators
op_a = OneParticleOperator("A", "electric")
op_b = OneParticleOperator("B", "electric")
op_c = OneParticleOperator("C", "electric")
op_d = OneParticleOperator("D", "electric")
op_e = OneParticleOperator("E", "electric")

# magnetic dipole operators
opm_a = OneParticleOperator("A", "magnetic")
opm_b = OneParticleOperator("B", "magnetic")
opm_c = OneParticleOperator("C", "magnetic")
opm_d = OneParticleOperator("D", "magnetic")
opm_e = OneParticleOperator("E", "magnetic")

# ADC matrix (for internal use)
M = Operator("M")

# diamagnetic magnetizability operators
xi_ab = OneParticleOperator("AB", "dia_magnet")
xi_bc = OneParticleOperator("BC", "dia_magnet")
xi_cd = OneParticleOperator("CD", "dia_magnet")

# electric quadrupole operators
Q_ab = OneParticleOperator("AB", "electric_quadrupole")
Q_bc = OneParticleOperator("BC", "electric_quadrupole")
Q_cd = OneParticleOperator("CD", "electric_quadrupole")
Q_de = OneParticleOperator("DE", "electric_quadrupole")
Q_ef = OneParticleOperator("EF", "electric_quadrupole")

# traceless electric quadrupole operators
theta_ab = OneParticleOperator("AB", "electric_quadrupole_traceless")
theta_bc = OneParticleOperator("BC", "electric_quadrupole_traceless")
theta_cd = OneParticleOperator("CD", "electric_quadrupole_traceless")
theta_de = OneParticleOperator("DE", "electric_quadrupole_traceless")
theta_ef = OneParticleOperator("EF", "electric_quadrupole_traceless")

# linear momentum operators
nabla_a = OneParticleOperator("A", "nabla")
nabla_b = OneParticleOperator("B", "nabla")
nabla_c = OneParticleOperator("C", "nabla")
nabla_d = OneParticleOperator("D", "nabla")
nabla_e = OneParticleOperator("E", "nabla")

