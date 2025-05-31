"""
Compute the excited-state polarizability according to Eq. (6) in 10.1063/5.0012120.
"""

import adcc
from pyscf import gto, scf

from responsefun import evaluate_property_isr, TransitionMoment
from responsefun.symbols_and_labels import f, gamma, n, mu_a, mu_b, w, w_f, w_n

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
print(state.describe())

# compute esp tensor
sos_expr = (
    TransitionMoment(f, mu_a, n) * TransitionMoment(n, mu_b, f) / (w_n - w_f - w - 1j*gamma)
    + TransitionMoment(f, mu_b, n) * TransitionMoment(n, mu_a, f) / (w_n - w_f + w + 1j*gamma)
)
tens = evaluate_property_isr(
    state,  # ExcitedStates object returned by the adcc calculation
    sos_expr,  # symbolic SOS expression
    [n],  # indices of summation
    excluded_states=f,  # state excluded from summation for convergence in the static limit
    freqs_in=[(w, 0.1)],  # incident frequencies
    freqs_out=[(w, 0.1)],  # incident frequencies
    damping=0.001,  # damping parameter
    excited_state=0,  # excited state f (0 corresponds to the first excited state)
    conv_tol=1e-4,
)

print(tens)