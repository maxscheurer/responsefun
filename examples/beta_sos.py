"""
Create a SumOverStates object using the first-order hyperpolarizability as an example.
"""

from responsefun.SumOverStates import SumOverStates, TransitionMoment
from responsefun.symbols_and_labels import (
    O,
    n,
    op_a,
    op_b,
    op_c,
    p,
    w_1,
    w_2,
    w_n,
    w_o,
    w_p,
)

beta_sos_term = (
    TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, p)
    * TransitionMoment(p, op_c, O) / ((w_n - w_o) * (w_p - w_2))
)

beta_sos = SumOverStates(
    beta_sos_term,  # first SOS term
    [n, p],  # indices of summation
    correlation_btw_freq=[(w_o, w_1+w_2)],  # correlation between the frequencies
    perm_pairs=[(op_a, -w_o), (op_b, w_1), (op_c, w_2)]  # tuples to be permuted
)

print("number of terms: {}".format(beta_sos.number_of_terms))
print(beta_sos.latex)
