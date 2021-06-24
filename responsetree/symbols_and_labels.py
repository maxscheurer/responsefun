from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex, simplify, fraction
from responsetree.response_operators import MTM, S2S_MTM, ResponseVector, Matrix, DipoleOperator


O, f = symbols(r"0, f", real=True)
gamma = symbols(r"\gamma", real=True)
n, m, p, k = symbols(r"n, m, p, k", real=True)
w_f, w_n, w_m, w_p, w_k = symbols(r"w_{f}, w_{n}, w_{m}, w_{p}, w_{k}", real=True)
w, w_o, w_1, w_2, w_3 = symbols(r"w, w_{\sigma}, w_{1}, w_{2}, w_{3}", real=True)

op_a = DipoleOperator("A")
op_b = DipoleOperator("B")
op_c = DipoleOperator("C")
op_d = DipoleOperator("D")

F_A = MTM("A")
F_B = MTM("B")
F_C = MTM("C")
F_D = MTM("D")

B_A = S2S_MTM("A")
B_B = S2S_MTM("B")
B_C = S2S_MTM("C")
B_D = S2S_MTM("D")

X_A = ResponseVector("A")
X_B = ResponseVector("B")
X_C = ResponseVector("C")
X_D = ResponseVector("D")

M = Matrix("M")
