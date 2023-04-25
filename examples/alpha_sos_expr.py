"""
Specify the SOS expression of a response function symbolically using the example of the linear polarizability.
"""

from sympy import symbols
from responsefun.ResponseOperator import (
    TransitionFrequency
)
from responsefun.symbols_and_labels import (
    O, f, gamma, n, m, p, k,
    op_a, op_b, op_c, op_d,
    w_f, w_n, w_m, w_p, w_k, w, w_o, w_1, w_2, w_3,
)
from responsefun.SumOverStates import TransitionMoment

# define symbols and operators
# O, n, gamma, w = symbols(r"0, n, \gamma, w", real=True)
# op_a = DipoleOperator("A")
# op_b = DipoleOperator("B")
# w_n = TransitionFrequency("n", real=True)

# define symbolic SOS expression
alpha_sos_expr = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w + 1j*gamma)
)

print(alpha_sos_expr)
