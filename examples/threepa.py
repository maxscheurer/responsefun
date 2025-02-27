"""
Compute three-photon absorption matrix element according to Eq. (5.252) in 10.1002/9781118794821.
"""
import adcc
import numpy as np
from pyscf import gto, scf

from responsefun.evaluate_property import evaluate_property_isr
from responsefun.SumOverStates import TransitionMoment
from responsefun.symbols_and_labels import (
    O,
    f,
    m,
    n,
    op_a,
    op_b,
    op_c,
    w_1,
    w_2,
    w_3,
    w_f,
    w_m,
    w_n,
)


def threepa_average(tens):
    assert np.shape(tens) == (3, 3, 3)
    return (1/35) * (2*np.einsum("abc,abc->", tens, tens) + 3*np.einsum("aab,bcc->", tens, tens))


# run SCF in PySCF
mol = gto.M(
    atom="""
    F       -0.000000   -0.000000    0.092567
    H        0.000000    0.000000   -0.833107
    """,
    unit="Angstrom",
    basis="augccpvdz",
)
scfres = scf.RHF(mol)
scfres.kernel()

# run ADC calculation using adcc
state = adcc.run_adc(scfres, method="adc2", n_singlets=10)
print(state.describe())

threepa_term = (
    TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m)
    * TransitionMoment(m, op_c, f) / ((w_n - w_1) * (w_m - w_1 - w_2))
)

for es in range(5):
    print(f"===== State {es} ===== ")
    # the minus sign is needed, because the negative charge is not yet included in the operator definitions
    # TODO: remove minus after adc-connect/adcc#190 is merged
    threepa_tens = -1.0 * evaluate_property_isr(
        state, threepa_term, [n, m],
        perm_pairs=[(op_a, w_1), (op_b, w_2), (op_c, w_3)],
        incoming_freqs=[(w_1, w_f/3), (w_2, w_f/3), (w_3, w_f/3)],
        excited_state=es, conv_tol=1e-5,
    )
    threepa_strength = threepa_average(threepa_tens)
    print(f"Transition strength (a.u.): {threepa_strength:.6f}")
