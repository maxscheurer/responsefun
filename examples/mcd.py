"""
Compute the Faraday MCD B term according to Eq. (4) in 10.1063/5.0013398
(taking into account the different definition of the transition moments).
"""
import adcc
import numpy as np
from pyscf import gto, scf

from responsefun.evaluate_property import evaluate_property_isr
from responsefun.misc import epsilon
from responsefun.SumOverStates import TransitionMoment
from responsefun.symbols_and_labels import O, j, k, op_a, op_b, opm_c, w_j, w_k

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
        TransitionMoment(O, opm_c, k) * TransitionMoment(k, op_b, j, shifted=True) * TransitionMoment(j, op_a, O)
        / w_k
)
mcd_sos_expr2 = (
        TransitionMoment(O, op_b, k) * TransitionMoment(k, opm_c, j) * TransitionMoment(j, op_a, O)
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

# the minus sign is needed, because the negative charge is not yet included in the operator definitions
# TODO: remove minus after adc-connect/adcc#190 is merged
bterm = -1.0 * np.einsum("abc,abc->", epsilon, mcd_tens1 + mcd_tens2)
print(f"The MCD Faraday B term for excited state {final_state} is {bterm:.2f} (a.u.).")