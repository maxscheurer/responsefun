"""
Specify the SOS expression of a response function symbolically using the example of the linear polarizability.
"""

from sympy import symbols
from responsefun.response_operators import (
    DipoleOperator,
    TransitionFrequency
)
from responsefun.sum_over_states import TransitionMoment

# define symbols and operators
O, n, gamma, w = symbols(r"0, n, \gamma, w", real=True)
op_a = DipoleOperator("A")
op_b = DipoleOperator("B")
w_n = TransitionFrequency("n", real=True)

# define symbolic SOS expression
alpha_sos_expr = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w + 1j*gamma)
)

print(alpha_sos_expr)
