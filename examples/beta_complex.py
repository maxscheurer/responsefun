from pyscf import gto, scf
import adcc
from responsefun.symbols_and_labels import (
    op_a, op_b, op_c,
    O, n, p, w_n, w_p,
    w_1, w_2, w_o, gamma
)
from responsefun.sum_over_states import TransitionMoment
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

# compute the complex beta tensor
beta_term = (
    TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, p) * TransitionMoment(p, op_c, O)
    / ((w_n - w_o - 1j*gamma) * (w_p - w_2 - 1j*gamma))
)
beta_tens = evaluate_property_isr(
    state, beta_term, [n, p], omegas=[(w_o, w_1+w_2+1j*gamma), (w_1, 0.5), (w_2, 0.5)], gamma_val=0.01,
    perm_pairs=[(op_a, -w_o-1j*gamma), (op_b, w_1+1j*gamma), (op_c, w_2+1j*gamma)]
)

print(beta_tens)
