from responsetree.symbols_and_labels import *
from responsetree.sum_over_states import TransitionMoment, SumOverStates
from responsetree.isr_conversion import to_isr
from responsetree.create_tree import build_tree

alpha_terms = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w + 1j*gamma)
)
alpha_sos = SumOverStates(alpha_terms, [n])
alpha_isr = to_isr(alpha_sos)
#print(alpha_sos.expr)
#print(alpha_isr)
#build_tree(alpha_isr)

rixs_terms = (
    TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
    + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w - w_f + 1j*gamma)
)
rixs_sos = SumOverStates(rixs_terms, [n])
rixs_isr = to_isr(rixs_sos)
#print(rixs_sos.expr)
#print(rixs_isr)
#build_tree(rixs_isr)

rixs_term_short = rixs_terms.args[0]
rixs_sos_short = SumOverStates(rixs_term_short, [n])
rixs_isr_short = to_isr(rixs_sos_short)
#print(rixs_sos_short.expr)
#print(rixs_isr_short)
#build_tree(rixs_isr_short)

tpa_terms = (
    TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - (w_f/2))
    + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - (w_f/2))
)
tpa_sos = SumOverStates(tpa_terms, [n])
tpa_isr = to_isr(tpa_sos)
#print(tpa_sos.expr)
#print(tpa_isr)
#build_tree(tpa_isr)

esp_terms = (
    TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - w_f - w - 1j*gamma)
    + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - w_f + w + 1j*gamma)
)
esp_sos = SumOverStates(esp_terms, [n])
esp_isr = to_isr(esp_sos)
#print(esp_sos.expr)
#print(esp_isr)
#build_tree(esp_isr)

beta_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o) * (w_k - w_2))
beta_sos = SumOverStates(beta_term, [n, k], [(w_o, w_1+w_2)], [(op_a, -w_o), (op_b, w_1), (op_c, w_2)])
beta_isr = to_isr(beta_sos)
#print(beta_sos.expr)
#print(beta_isr)
#build_tree(beta_isr)


#TODO: make it work for threepa and gamma

threepa_term = TransitionMoment(O, op_b, m) * TransitionMoment(m, op_c, n) * TransitionMoment(n, op_d, f) / ((w_n - w_1 - w_2) * (w_m - w_1))
threepa_sos = SumOverStates(threepa_term, [m, n], [(w_f, w_1+w_2+w_3)], [(op_b, w_1), (op_c, w_2), (op_d, w_3)])
#threepa_isr = to_isr(threepa_sos)
#print(threepa_sos.expr)
#print(len(threepa_isr.args))

gamma_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m) * TransitionMoment(m, op_c, k) * TransitionMoment(k, op_d, O) / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_k - w_3))
gamma_sos = SumOverStates(gamma_term, [n, m, k], [(w_o, w_1+w_2+w_3)], [(op_a, -w_o), (op_b, w_1), (op_c, w_2), (op_d, w_3)])
#gamma_isr = to_isr(gamma_sos)
#print(gamma_sos.expr)
#print(gamma_isr)
