from sympy import symbols
from sympy.physics.quantum.operator import Operator

from responsefun.ResponseOperator import OneParticleOperator, TransitionFrequency

# ground state and excited state f
O, f, j = symbols("O, f, j", real=True)

# damping factor
gamma = symbols("gamma", real=True)

# indices of summation
n, m, p, k = symbols("n, m, p, k", real=True)

# external frequencies
w, w_o, w_1, w_2, w_3, w_prime = symbols("w, w_o, w_1, w_2, w_3, w'", real=True)

# transition frequencies
w_f = TransitionFrequency(f, real=True)
w_j = TransitionFrequency(j, real=True)
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
