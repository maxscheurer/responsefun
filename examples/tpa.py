"""
Compute two-photon absorption matrix element according to Eq. (5.250) in 10.1002/9781118794821.
"""
import adcc
import numpy as np
from pyscf import gto, scf

from responsefun.evaluate_property import evaluate_property_isr
from responsefun.SumOverStates import TransitionMoment
from responsefun.symbols_and_labels import (
    O,
    f,
    n,
    op_a,
    op_b,
    w_1,
    w_2,
    w_n,
    w_f,
)


def tpa_average(tens):
    assert np.shape(tens) == (3, 3)
    return (1/15) * (2*np.einsum("ab,ab->", tens, tens) + np.einsum("aa,bb->", tens, tens))


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
state = adcc.run_adc(scfres, method="adc2", n_singlets=5)
print(state.describe())

tpa_term = (
    TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, O) / (w_n - w_1)
)

for es in range(1):
    print(f"===== State {es} ===== ")
    tpa_tens = evaluate_property_isr(
        state, tpa_term, [n],
        perm_pairs=[(op_a, w_1), (op_b, w_2)],
        incoming_freqs=[(w_1, w_f/2), (w_2, w_f/2)],
        excited_state=es, conv_tol=1e-4,
    )
    tpa_strength = tpa_average(tpa_tens)
    print(f"Transition strength (a.u.): {tpa_strength:.6f}")