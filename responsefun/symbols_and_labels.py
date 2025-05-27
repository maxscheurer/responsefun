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
mu_a = OneParticleOperator("A", "electric_dipole", False)
mu_b = OneParticleOperator("B", "electric_dipole", False)
mu_c = OneParticleOperator("C", "electric_dipole", False)
mu_d = OneParticleOperator("D", "electric_dipole", False)
mu_e = OneParticleOperator("E", "electric_dipole", False)

# electric dipole operators in velocity gauge
mup_a = OneParticleOperator("A", "electric_dipole_velocity", False)
mup_b = OneParticleOperator("B", "electric_dipole_velocity", False)
mup_c = OneParticleOperator("C", "electric_dipole_velocity", False)
mup_d = OneParticleOperator("D", "electric_dipole_velocity", False)
mup_e = OneParticleOperator("E", "electric_dipole_velocity", False)

# magnetic dipole operators
m_a = OneParticleOperator("A", "magnetic_dipole", False)
m_b = OneParticleOperator("B", "magnetic_dipole", False)
m_c = OneParticleOperator("C", "magnetic_dipole", False)
m_d = OneParticleOperator("D", "magnetic_dipole", False)
m_e = OneParticleOperator("E", "magnetic_dipole", False)

# electric quadrupole operators
q_ab = OneParticleOperator("AB", "electric_quadrupole", False)
q_bc = OneParticleOperator("BC", "electric_quadrupole", False)
q_cd = OneParticleOperator("CD", "electric_quadrupole", False)
q_de = OneParticleOperator("DE", "electric_quadrupole", False)
q_ef = OneParticleOperator("EF", "electric_quadrupole", False)

# electric quadrupole operators in velocity gauge
qp_ab = OneParticleOperator("AB", "electric_quadrupole_velocity", False)
qp_bc = OneParticleOperator("BC", "electric_quadrupole_velocity", False)
qp_cd = OneParticleOperator("CD", "electric_quadrupole_velocity", False)
qp_de = OneParticleOperator("DE", "electric_quadrupole_velocity", False)
qp_ef = OneParticleOperator("EF", "electric_quadrupole_velocity", False)

# diamagnetic magnetizability operators
xi_ab = OneParticleOperator("AB", "diamagnetic_magnetizability", False)
xi_bc = OneParticleOperator("BC", "diamagnetic_magnetizability", False)
xi_cd = OneParticleOperator("CD", "diamagnetic_magnetizability", False)
xi_de = OneParticleOperator("DE", "diamagnetic_magnetizability", False)
xi_ef = OneParticleOperator("EF", "diamagnetic_magnetizability", False)