"""
Create a SumOverStates object using the first-order hyperpolarizability as an example.
"""

from responsefun.SumOverStates import SumOverStates, TransitionMoment
from responsefun.symbols_and_labels import (
    O,
    n,
    mu_a,
    mu_b,
    mu_c,
    p,
    w_1,
    w_2,
    w_n,
    w_o,
    w_p,
)

beta_sos_term = (
    TransitionMoment(O, mu_a, n) * TransitionMoment(n, mu_b, p, shifted=True)
    * TransitionMoment(p, mu_c, O) / ((w_n - w_o) * (w_p - w_2))
)

beta_sos = SumOverStates(
    beta_sos_term,  # first SOS term
    [n, p],  # indices of summation
    freqs_in=[w_1, w_2],  # frequencies of incident photons
    freqs_out=w_o,  # frequency of resulting photon
    perm_pairs=[(mu_a, -w_o), (mu_b, w_1), (mu_c, w_2)],  # tuples to be permuted
    excluded_states=O  # states excluded from the summations
)

print("number of terms: {}".format(beta_sos.number_of_terms))
print(beta_sos)
print(f"energy balance: {beta_sos.energy_balance}")
print(f"found correlation between frequencies: {beta_sos.correlation_btw_freq}")
print(f"For Latex:\n{beta_sos.latex}")