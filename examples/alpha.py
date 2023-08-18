"""
Compute the linear polarizability in the static limit for water using the STO-3G basis set.
"""
from pyscf import gto, scf
import adcc
import numpy as np
from responsefun.symbols_and_labels import (
    O, gamma, n, op_a, op_b, w_n, w
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

# define symbolic SOS expression
alpha_sos_expr = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w + 1j*gamma)
)
# compute polarizability
alpha_tens = evaluate_property_isr(
    state, alpha_sos_expr, [n], omegas=[(w, 0)], conv_tol=1e-5
)
print(alpha_tens)
aver = (1/3) * np.trace(alpha_tens)
print(f"The isotropic average is {aver:.2f} (a.u.).")


