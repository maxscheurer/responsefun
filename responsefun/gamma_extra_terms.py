import numpy as np
from itertools import permutations, product
import adcc
from pyscf import gto, scf


def sos_antonia(state, omega_1, omega_2, omega_3):
    omega_o = omega_1 + omega_2 + omega_3
    sos = np.zeros((3, 3, 3, 3))
    components = list(product([0, 1, 2], repeat=4))
    for n, dip_n in enumerate(state.transition_dipole_moment):
        for m, dip_m in enumerate(state.transition_dipole_moment):
            for c in components:
                A, B, C, D = c
                perms = list(permutations([(A, -omega_o), (B, omega_1), (C, omega_2), (D, omega_3)]))
                for p in perms:
                    sos[c] += dip_n[p[0][0]]*dip_n[p[1][0]]*dip_m[p[2][0]]*dip_m[p[3][0]] / ( 
                            (state.excitation_energy_uncorrected[n] + p[0][1])
                            *(p[2][1] + p[3][1])
                            *(state.excitation_energy_uncorrected[m] - p[3][1])
                    )
    return sos


def sos_panor(state, omega_1, omega_2, omega_3):
    omega_o = omega_1 + omega_2 + omega_3
    sos = np.zeros((3, 3, 3, 3))
    components = list(product([0, 1, 2], repeat=4))
    for n, dip_n in enumerate(state.transition_dipole_moment):
        for m, dip_m in enumerate(state.transition_dipole_moment):
            for c in components:
                A, B, C, D = c
                perms = list(permutations([(A, -omega_o), (B, omega_1), (C, omega_2), (D, omega_3)]))
                for p in perms:
                    sos[c] += dip_n[p[0][0]]*dip_n[p[1][0]]*dip_m[p[2][0]]*dip_m[p[3][0]] / (
                            (state.excitation_energy_uncorrected[n] + p[0][1])
                            *(state.excitation_energy_uncorrected[m] - p[3][1])
                            *(state.excitation_energy_uncorrected[m] + p[2][1])
                    )
    return sos


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
scfres.conv_tol = 1e-8
scfres.conv_tol_grad = 1e-8
scfres.kernel()

refstate = adcc.ReferenceState(scfres)
matrix = adcc.AdcMatrix("adc2", refstate)
state = adcc.adc2(scfres, n_singlets=65)
#print(state.describe())

panor_terms = sos_panor(state, 0.0, 0.0, 0.0)
#antonia_terms = sos_antonia(state, 0.5, 0.3, 0.0)
print(panor_terms)
#print(antonia_terms)
#np.testing.assert_allclose(panor_terms, antonia_terms, atol=1e-7)
