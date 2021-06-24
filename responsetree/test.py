from responsetree.symbols_and_labels import *
from responsetree.transition_moments import TransitionMoment
from responsetree.sum_over_states import SumOverStates
from responsetree.isr_conversion import to_isr

alpha_term = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w + 1j*gamma)
    )
alpha_sos = SumOverStates(alpha_term, [n])
#print(alpha_sos.expr)
alpha_isr = to_isr(alpha_sos.expr, alpha_sos.summation_indices, alpha_sos.operators)
#print(alpha_isr)

rixs_term = (
    TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
    + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w - w_f + 1j*gamma)
)
rixs_sos = SumOverStates(rixs_term, [n])
#print(rixs_sos.expr)
rixs_isr = to_isr(rixs_sos.expr, rixs_sos.summation_indices, rixs_sos.operators)
#print(rixs_isr)

rixs_term_short = rixs_term.args[0]
rixs_sos_short = SumOverStates(rixs_term_short, [n])
#print(rixs_sos_short.expr)
rixs_isr_short = to_isr(rixs_sos_short.expr, rixs_sos_short.summation_indices, rixs_sos_short.operators)
#print(rixs_isr_short)

tpa_term = (
    TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - (w_f/2))
    + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - (w_f/2))
)
tpa_sos = SumOverStates(tpa_term, [n])
#print(tpa_sos.expr)
tpa_isr = to_isr(tpa_sos.expr, tpa_sos.summation_indices)
#print(tpa_isr)

esp_term = (
    TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - w_f - w - 1j*gamma)
    + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - w_f + w + 1j*gamma)
)
esp_sos = SumOverStates(esp_term, [n])
#print(esp_sos.expr)
esp_isr = to_isr(esp_sos.expr, esp_sos.summation_indices)
#print(esp_isr)

beta_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o) * (w_k - w_2))
beta_sos = SumOverStates(beta_term, [n, k], [(w_o, w_1+w_2)], [(op_a, -w_o), (op_b, w_1), (op_c, w_2)])
#print(beta_sos.expr)
beta_isr = to_isr(beta_sos.expr, beta_sos.summation_indices, beta_sos.operators, beta_sos.correlation_btw_freq)
#print(beta_isr)

gamma_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m) * TransitionMoment(m, op_c, k) * TransitionMoment(k, op_d, O) / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_k - w_3))
gamma_sos = SumOverStates(gamma_term, [n, m, k])
