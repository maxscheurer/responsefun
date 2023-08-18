"""
Compute the Faraday MCD B term for water using the STO-3G basis set.
Compare 
"""
from pyscf import gto, scf
import adcc
import numpy as np
from responsefun.symbols_and_labels import (
    O, k, j, opm_b, op_c, op_a, w_k, w_j
)
from responsefun.SumOverStates import TransitionMoment
from responsefun.evaluate_property import evaluate_property_isr

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
        TransitionMoment(O, opm_b, k) * TransitionMoment(k, op_c, j) * TransitionMoment(j, op_a, O) / w_k
)
mcd_sos_expr2 = (
        TransitionMoment(O, op_c, k) * TransitionMoment(k, opm_b, j) * TransitionMoment(j, op_a, O) / (w_k - w_j)

)
final_state = 0
# compute MCD B term
mcd_tens1 = evaluate_property_isr(
    state, mcd_sos_expr1, [k], final_state=(j, final_state), excluded_states=O, conv_tol=1e-5
)
mcd_tens2 = evaluate_property_isr(
    state, mcd_sos_expr2, [k], final_state=(j, final_state), excluded_states=j, conv_tol=1e-5
)

# Levi-Civita tensor
epsilon = np.zeros((3, 3, 3))
epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
epsilon[2, 1, 0] = epsilon[0, 2, 1] = epsilon[1, 0, 2] = -1

bterm = np.einsum("abc,abc->", epsilon, mcd_tens1 + mcd_tens2)
print(f"The MCD Faraday B term for excited state {final_state} is {bterm:.2f} (a.u.).")


