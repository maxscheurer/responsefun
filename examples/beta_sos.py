"""
Create a SumOverStates object using the first-order hyperpolarizability as an example.
"""

from responsefun.symbols_and_labels import (
    op_a, op_b, op_c, O, n, p, w_n, w_p, w_o, w_1, w_2
)
from responsefun.SumOverStates import TransitionMoment, SumOverStates

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
