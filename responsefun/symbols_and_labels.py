from sympy import symbols
from sympy.physics.quantum.operator import Operator
from responsefun.ResponseOperator import (
    MTM, S2S_MTM, ResponseVector, OneParticleOperator, TransitionFrequency
)


O, f = symbols("O, f", real=True)
gamma = symbols("gamma", real=True)
n, m, p, k = symbols("n, m, p, k", real=True)
w, w_o, w_1, w_2, w_3 = symbols("w, w_sigma, w_1, w_2, w_3", real=True)

w_f = TransitionFrequency(f, real=True)
w_n = TransitionFrequency(n, real=True)
w_m = TransitionFrequency(m, real=True)
w_p = TransitionFrequency(p, real=True)
w_k = TransitionFrequency(k, real=True)

op_a = OneParticleOperator("A", "electric")
op_b = OneParticleOperator("B", "electric")
op_c = OneParticleOperator("C", "electric")
op_d = OneParticleOperator("D", "electric")
op_e = OneParticleOperator("E", "electric")

opm_a = OneParticleOperator("A", "magnetic")
opm_b = OneParticleOperator("B", "magnetic")
opm_c = OneParticleOperator("C", "magnetic")
opm_d = OneParticleOperator("D", "magnetic")
opm_e = OneParticleOperator("E", "magnetic")

F_A = MTM("A", "electric")
F_B = MTM("B", "electric")
F_C = MTM("C", "electric")
F_D = MTM("D", "electric")

B_A = S2S_MTM("A", "electric")
B_B = S2S_MTM("B", "electric")
B_C = S2S_MTM("C", "electric")
B_D = S2S_MTM("D", "electric")

X_A = ResponseVector("A")
X_B = ResponseVector("B")
X_C = ResponseVector("C")
X_D = ResponseVector("D")

M = Operator("M")
