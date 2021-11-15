import numpy as np

from pyscf import gto, scf
import adcc

def alpha_sos(state, omega):
    omega_1 = 1/(state.excitation_energy_uncorrected - omega)
    omega_2 = 1/(state.excitation_energy_uncorrected + omega)
    einsum_string = "nA, nB, n -> AB"
    array_list = [state.transition_dipole_moment, state.transition_dipole_moment, omega_1]
    print(array_list)
    ret = (
        np.einsum(einsum_string, *array_list)
        + np.einsum('nB, nA, n -> BA', state.transition_dipole_moment, state.transition_dipole_moment, omega_2)
    )
    return ret

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

refstate = adcc.ReferenceState(scfres)
matrix = adcc.AdcMatrix("adc2", refstate)
state = adcc.adc2(scfres, n_singlets=65)

omega = 0.59
alpha = alpha_sos(state, omega)
print(alpha)
