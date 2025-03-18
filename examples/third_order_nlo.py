"""
Compute third-order nonlinear optical properties (see 10.1021/acs.jctc.3c00456)
with an SOS expression for the second-order hyperpolarizability according to
Eq. (5.201) in 10.1002/9781118794821.
"""
import adcc
from pyscf import gto, scf
import numpy as np

from responsefun import evaluate_property_isr, TransitionMoment
from responsefun.symbols_and_labels import (
    O,
    m,
    n,
    mu_a,
    mu_b,
    mu_c,
    mu_d,
    p,
    w_1,
    w_2,
    w_3,
    w_m,
    w_n,
    w_o,
    w_p,
)


def compute_gamma_average(gamma_tens):
    gamma_aver = (1/15) * (
        np.einsum("iijj->", gamma_tens)
        + np.einsum("ijij", gamma_tens)
        + np.einsum("ijji", gamma_tens)
    )
    return gamma_aver


# run SCF in PySCF
mol = gto.M(
    atom="""
    O        0.000000    0.000000    0.115082
    H        0.000000    0.767545   -0.460329
    H        0.000000   -0.767545   -0.460329
    """,
    unit="Angstrom",
    basis="aug-cc-pvdz",
)
scfres = scf.RHF(mol)
scfres.kernel()

w_ruby = 0.0656

# run ADC(2) calculation using adcc
state = adcc.adc2(scfres, n_singlets=5)
# compute the second hyperpolarizability tensor
gamma_term_I = (
    TransitionMoment(O, mu_a, n) * TransitionMoment(n, mu_b, m, shifted=True)
    * TransitionMoment(m, mu_c, p, shifted=True) * TransitionMoment(p, mu_d, O)
    / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
)
gamma_term_II = (
    TransitionMoment(O, mu_a, n) * TransitionMoment(n, mu_b, O)
    * TransitionMoment(O, mu_c, m) * TransitionMoment(m, mu_d, O)
    / ((w_n - w_o) * (w_m - w_3) * (w_m + w_2))
)
perm_pairs = [(mu_a, -w_o), (mu_b, w_1), (mu_c, w_2), (mu_d, w_3)]
processes = {
    "static": (0.0, 0.0, 0.0), "dcOR": (w_ruby, -w_ruby, 0.0),
    "EOKE": (w_ruby, 0.0, 0.0), "IDRI": (w_ruby, -w_ruby, w_ruby),
    "ESHG": (w_ruby, w_ruby, 0.0), "THG": (w_ruby, w_ruby, w_ruby)
}
for process, freqs in processes.items():
    freqs_in = [(w_1, freqs[0]), (w_2, freqs[1]), (w_3, freqs[2])]
    freqs_out = (w_o, w_1+w_2+w_3)
    gamma_tens_I = evaluate_property_isr(
        state, gamma_term_I, [n, m, p],
        perm_pairs=perm_pairs, excluded_states=O,
        freqs_in=freqs_in, freqs_out=freqs_out,
        conv_tol=1e-5,
    )
    gamma_tens_II = evaluate_property_isr(
        state, gamma_term_II, [n, m],
        perm_pairs=perm_pairs, excluded_states=O,
        freqs_in=freqs_in, freqs_out=freqs_out,
        conv_tol=1e-5,
    )
    gamma_tens = gamma_tens_I - gamma_tens_II
    print(process)
    print(gamma_tens)
    print(f"gamma_average = {compute_gamma_average(gamma_tens):.2f} a.u.")
