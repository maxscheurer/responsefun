"""
Compute the excited-state polarizability for water using the STO-3G basis set.
"""

from pyscf import gto, scf
import adcc
from responsefun.symbols_and_labels import (
    op_a, op_b, f, n, gamma, w_n, w_f, w
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
print(state.describe())

# compute esp tensor
sos_expr = (
    TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - w_f - w - 1j*gamma)
    + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - w_f + w + 1j*gamma)
)
tens = evaluate_property_isr(
    state,  # ExcitedStates object returned by the adcc calculation
    sos_expr,  # symbolic SOS expression
    [n],  # indices of summation
    omegas=[(w, 0.1)],  # incident frequencies
    gamma_val=0.001,  # damping parameter
    final_state=(f, 0),  # excited state f (0 corresponds to the first excited state)
    excluded_states=f  # state excluded from summation for convergence in the static limit
)

print(tens)