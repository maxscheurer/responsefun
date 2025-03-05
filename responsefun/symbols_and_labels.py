from sympy import symbols

from responsefun.operators import OneParticleOperator, TransitionFrequency

# ground state and excited state f
O, f, j = symbols("O, f, j", real=True)

# damping factor
gamma = symbols("gamma", real=True)

# indices of summation
n, m, p, k = symbols("n, m, p, k", real=True)

# external frequencies
w, w_o, w_1, w_2, w_3, w_4, w_prime = symbols("w, w_o, w_1, w_2, w_3, w_4, w'", real=True)
w_a, w_b, w_c = symbols("w_a, w_b, w_c", real=True)

# transition frequencies
w_f = TransitionFrequency(f, real=True)
w_j = TransitionFrequency(j, real=True)
w_n = TransitionFrequency(n, real=True)
w_m = TransitionFrequency(m, real=True)
w_p = TransitionFrequency(p, real=True)
w_k = TransitionFrequency(k, real=True)

# electric dipole operators
op_a = OneParticleOperator("A", "electric_dipole", False)
op_b = OneParticleOperator("B", "electric_dipole", False)
op_c = OneParticleOperator("C", "electric_dipole", False)
op_d = OneParticleOperator("D", "electric_dipole", False)
op_e = OneParticleOperator("E", "electric_dipole", False)

# magnetic dipole operators
opm_a = OneParticleOperator("A", "magnetic_dipole", False)
opm_b = OneParticleOperator("B", "magnetic_dipole", False)
opm_c = OneParticleOperator("C", "magnetic_dipole", False)
opm_d = OneParticleOperator("D", "magnetic_dipole", False)
opm_e = OneParticleOperator("E", "magnetic_dipole", False)