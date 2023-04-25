"""
Specify the SOS expression of a response function symbolically using the example of the linear polarizability.
"""

from responsefun.symbols_and_labels import (
    O, gamma, n, op_a, op_b, w_n, w
)
from responsefun.SumOverStates import TransitionMoment

# define symbolic SOS expression
alpha_sos_expr = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w + 1j*gamma)
)

print(alpha_sos_expr)
