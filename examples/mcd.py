"""
Compute the Faraday MCD B term according to Eq. (4) in 10.1063/5.0013398
(taking into account the different definition of the transition moments).
"""
import adcc
import numpy as np
from pyscf import gto, scf

from responsefun import evaluate_property_isr, TransitionMoment
from responsefun.misc import epsilon
from responsefun.symbols_and_labels import O, j, k, mu_a, mu_b, m_c, w_j, w_k

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

# define symbolic SOS expressions
mcd_sos_expr1 = (
        TransitionMoment(O, m_c, k) * TransitionMoment(k, mu_b, j, shifted=True)
        * TransitionMoment(j, mu_a, O) / w_k
)
mcd_sos_expr2 = (
        TransitionMoment(O, mu_b, k) * TransitionMoment(k, m_c, j) * TransitionMoment(j, mu_a, O)
        / (w_k - w_j)
)
# compute MCD B term for the first excited state
final_state = 0
mcd_tens1 = evaluate_property_isr(
    state, mcd_sos_expr1, [k],
    excluded_states=O, excited_state=final_state,
    conv_tol=1e-4,
)
mcd_tens2 = evaluate_property_isr(
    state, mcd_sos_expr2, [k],
    excluded_states=[O,j], excited_state=final_state,
    conv_tol=1e-4,
)

bterm = np.einsum("abc,abc->", epsilon, mcd_tens1 + mcd_tens2)
print(f"The MCD Faraday B term for excited state {final_state} is {bterm:.2f} (a.u.).")