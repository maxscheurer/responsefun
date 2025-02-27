"""
Compute RIXS amplitudes according to Eq. (1) in 10.1021/acs.jctc.7b00636.
"""

import adcc
from pyscf import gto, scf
from scipy.constants import physical_constants

from responsefun.evaluate_property import evaluate_property_isr
from responsefun.SumOverStates import TransitionMoment
from responsefun.symbols_and_labels import (
    O,
    f,
    gamma,
    n,
    op_a,
    op_b,
    w,
    w_f,
    w_n,
    w_prime,
)


def ev2au(ev):
    return ev / physical_constants["hartree-electron volt relationship"][0]


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

# compute RIXS tensor within the rotating-wave approximation
rixs_sos_rwa = TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j * gamma)
rixs_rwa = evaluate_property_isr(
    state,
    rixs_sos_rwa,
    [n],
    freqs_in=(w, ev2au(534.74)),
    freqs_out=(w_prime, w-w_f),
    damping=ev2au(0.124),
    excited_state=2,
    conv_tol=1e-4,
)

# compute full RIXS tensor
rixs_full = evaluate_property_isr(
    state,
    rixs_sos_rwa,
    [n],
    perm_pairs=[(op_a, w + 1j * gamma), (op_b, -w_prime - 1j * gamma)],
    freqs_in=(w, ev2au(534.74)),
    freqs_out=(w_prime, w-w_f),
    damping=ev2au(0.124),
    excited_state=2,
    conv_tol=1e-4,
)

print(f"RIXS tensor within the rotating-wave approximation:\n{rixs_rwa}")
print(f"Full RIXS tensor:\n{rixs_full}")