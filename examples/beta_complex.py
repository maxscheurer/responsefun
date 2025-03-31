"""
Compute first-order hyperpolarizability according to Eq. (5.310) in 10.1002/9781118794821.
"""
import adcc
from pyscf import gto, scf

from responsefun import evaluate_property_isr, TransitionMoment
from responsefun.symbols_and_labels import (
    O,
    gamma,
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

# run SCF in PySCF
mol = gto.M(
    atom="""
    O 0 0 0
    H 0 0 1.795239827225189
    H 1.693194615993441 0 -0.599043184453037
    """,
    unit="Bohr",
    basis="sto-3g",
)
scfres = scf.RHF(mol)
scfres.kernel()

# run ADC calculation using adcc
state = adcc.adc2(scfres, n_singlets=5)

# compute the complex beta tensor
beta_term = (
    TransitionMoment(O, mu_a, n) * TransitionMoment(n, mu_b, p, shifted=True)
    * TransitionMoment(p, mu_c, O) / ((w_n - w_o - 1j*gamma) * (w_p - w_2 - 1j*gamma))
    + TransitionMoment(O, mu_b, n) * TransitionMoment(n, mu_a, p, shifted=True)
    * TransitionMoment(p, mu_c, O) / ((w_n + w_1 + 1j*gamma) * (w_p - w_2 - 1j*gamma))
    + TransitionMoment(O, mu_b, n) * TransitionMoment(n, mu_c, p, shifted=True)
    * TransitionMoment(p, mu_a, O) / ((w_n + w_1 + 1j*gamma) * (w_p + w_o + 1j*gamma))
)
beta_tens = evaluate_property_isr(
    state, beta_term, [n, p],
    perm_pairs=[(mu_b, w_1), (mu_c, w_2)], excluded_states=O,
    freqs_in=[(w_1, 0.5), (w_2, 0.5)], freqs_out=(w_o, w_1+w_2),
    damping=0.01, conv_tol=1e-4,
)

print(beta_tens)