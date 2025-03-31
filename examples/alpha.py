"""
Compute the linear polarizability in the static limit according to 10.1063/1.4977039.
"""
import adcc
import numpy as np
from pyscf import gto, scf

from responsefun import evaluate_property_isr, TransitionMoment
from responsefun.symbols_and_labels import O, gamma, n, mu_a, mu_b, w, w_n

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

# define symbolic SOS expression
alpha_sos_expr = (
        TransitionMoment(O, mu_a, n) * TransitionMoment(n, mu_b, O) / (w_n - w - 1j*gamma)
        + TransitionMoment(O, mu_b, n) * TransitionMoment(n, mu_a, O) / (w_n + w + 1j*gamma)
)
# compute polarizability
alpha_tens = evaluate_property_isr(
    state, alpha_sos_expr, [n], excluded_states=O,
    freqs_in=(w, 0), freqs_out=(w, 0),
    conv_tol=1e-4, 
)
print(alpha_tens)
aver = (1/3) * np.trace(alpha_tens)
print(f"The isotropic average is {aver:.2f} (a.u.).")


