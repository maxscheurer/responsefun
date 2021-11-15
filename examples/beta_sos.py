"""
Create a SumOverStates object using the first-order hyperpolarizability as an example.
"""

from responsefun.symbols_and_labels import *
from responsefun.sum_over_states import TransitionMoment, SumOverStates

beta_sos_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, p) * TransitionMoment(p, op_c, O) / ((w_n - w_o) * (w_p - w_2))

beta_sos = SumOverStates(
    beta_sos_term, # first SOS term
    [n, p], # indices of summation
    [(w_o, w_1+w_2)], # correlation between the frequencies
    [(op_a, -w_o), (op_b, w_1), (op_c, w_2)] # tuples to be permuted
)

print("number of terms: {}".format(beta_sos.number_of_terms))
print(beta_sos.expr)
