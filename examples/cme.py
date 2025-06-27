"""
Compute the Cotton-Mouton constant according to Eq. 2 in 10.1021/acs.jpca.3c04963
"""
import adcc
import numpy as np
from pyscf import gto, scf
from scipy import constants

from responsefun import evaluate_property_isr, TransitionMoment
from responsefun.symbols_and_labels import (O, n, p, m, mu_a, mu_b, m_a, m_b, m_c, m_d, xi_cd,
w_n, w_p, w_o, w_2, w_m, w_3, w, w_1
)
from responsefun.AdccProperties import DiamagneticMagnetizability

# run SCF in PySCF
mol = gto.M(
    atom="""
    O 0 0 0
    H 0 0 1.795239827225189
    H 1.693194615993441 0 -0.599043184453037
    """,
    unit="Bohr",
    basis="sto-3g"
)
scfres = scf.RHF(mol)
scfres.kernel()

# run ADC calculation using adcc
state = adcc.adc2(scfres, n_singlets=1)

# define (first term of) symbolic SOS expressions
alpha_sos_expr = (
        TransitionMoment(O, mu_a, n) * TransitionMoment(n, mu_b, O) / (w_n - w)
        + TransitionMoment(O, mu_b, n) * TransitionMoment(n, mu_a, O) / (w_n + w)
)
xi_para_sos_expr = (
        TransitionMoment(O, m_a, n) * TransitionMoment(n, m_b, O) / (w_n - w)
        + TransitionMoment(O, m_b, n) * TransitionMoment(n, m_a, O) / (w_n + w)
)
eta_dia_sos_expr = (
    TransitionMoment(O, mu_a, n) * TransitionMoment(n, mu_b, p, shifted=True)
    * TransitionMoment(p, xi_cd, O) / ((w_n - w_o) * (w_p - w_2))
)
eta_para_term_I_sos_expr = (
    TransitionMoment(O, mu_a, n) * TransitionMoment(n, mu_b, m, shifted=True)
    * TransitionMoment(m, m_c, p, shifted=True) * TransitionMoment(p, m_d, O)
    / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
)
eta_para_term_II_sos_expr = (
    TransitionMoment(O, mu_a, n) * TransitionMoment(n, mu_b, O)
    * TransitionMoment(O, m_c, m) * TransitionMoment(m, m_d, O)
    / ((w_n - w_o) * (w_m - w_3) * (w_m + w_2))
)

w_ruby = 0.072
gauge_origin = "mass_center"

# calculate tensors
alpha_tens = evaluate_property_isr(
    state, # ExcitedStates object returned by adcc calculation
    alpha_sos_expr, # symbolic SOS expression
    [n], # indices of summation
    excluded_states=O,# states excluded from summation (here: ground state)
    freqs_in=(w, w_ruby), # incoming frequencies
    freqs_out=(w, w_ruby), # outcoming frequencies
    conv_tol=1e-4, # convergence tolerance for response solver
    gauge_origin=gauge_origin # gauge origin for operator integrals
)

xi_para_tens = evaluate_property_isr(
    state, xi_para_sos_expr, [n], excluded_states=O,
    freqs_in=(w, 0), freqs_out=(w, 0),
    conv_tol=1e-4, 
    gauge_origin=gauge_origin
)

xi_dia_tens = DiamagneticMagnetizability(state, gauge_origin=gauge_origin).gs_moment

eta_dia_tens = evaluate_property_isr(
    state, eta_dia_sos_expr, [n, p],
    perm_pairs=[(mu_a, -w_o), (mu_b, w_1), (xi_cd, w_2)], # pairs to be permuted
    freqs_in=[(w_1, w_ruby), (w_2, 0)],
    freqs_out=(w_o, w_1+w_2),
    excluded_states=O,
    conv_tol=1e-4,
    gauge_origin=gauge_origin
)

eta_para_tens_I = evaluate_property_isr(
    state, eta_para_term_I_sos_expr, [n, m, p],
    perm_pairs=[(mu_a, -w_o), (mu_b, w_1), (m_c, w_2), (m_d, w_3)],
    excluded_states=O,
    freqs_in=[(w_1, w_ruby), (w_2, 0), (w_3, 0)],
    freqs_out=(w_o, w_1+w_2+w_3),
    conv_tol=1e-4,
    gauge_origin=gauge_origin
)

eta_para_tens_II = evaluate_property_isr(
    state, eta_para_term_II_sos_expr, [n, m],
    perm_pairs=[(mu_a, -w_o), (mu_b, w_1), (m_c, w_2), (m_d, w_3)],
    excluded_states=O,
    freqs_in=[(w_1, w_ruby), (w_2, 0), (w_3, 0)],
    freqs_out=(w_o, w_1+w_2+w_3),
    conv_tol=1e-4,
    gauge_origin=gauge_origin
)

# add dia- and paramagnetic contributions
xi_tot = xi_dia_tens + xi_para_tens
eta_tot = eta_dia_tens + (eta_para_tens_I - eta_para_tens_II)

# calculate anisotropies
alpha_aniso = alpha_tens[2, 2] - alpha_tens[0, 0]
xi_aniso = xi_tot[2, 2] - xi_tot[0, 0]
eta_aniso = 1/15 * (7 * eta_tot[0, 0, 0, 0] - 5 * eta_tot[0, 0, 1, 1] + 2 * eta_tot[2, 2, 2, 2]
                   - 2 * eta_tot[0, 0, 2, 2] - 2 * eta_tot[2, 2, 0, 0] + 12 * eta_tot[0, 2, 0, 2])

# compute CME constant in cgs system [cm^3 G^–2 mol^–1] at T=298.15K 
pi = constants.pi
N_A = constants.N_A
Hartree = constants.physical_constants['atomic unit of energy'][0]
k_B = constants.k/Hartree
T = 273.15
const_CME = ((2 * np.pi *N_A)/27 * (eta_aniso + 2/ (15 * k_B * T ) * alpha_aniso * xi_aniso))
const_CME_cgs = const_CME * 2.68211e-44 * 1e20
print(f"The Cotton-Mouton constant at 298.15 K is {const_CME_cgs:.2f} (10^-20 cm^3 G^–2 mol^–1).")
