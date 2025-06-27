"""
Compute the two-photon circular dichroism rotatory strength in the velocity gauge
according to 10.1021/acs.jpca.5c02108 eqs. 21-27.
"""
import adcc
import numpy as np
from pyscf import gto, scf

from responsefun import evaluate_property_isr, TransitionMoment
from responsefun.misc import epsilon
from responsefun.symbols_and_labels import (O, f, n, m_b, mup_a, mup_b, mup_c, 
                                            qp_ab, w_a, w_b, w_f, w_n)

# The calculation of two-photon circular dichroism requires three different two-photon tensors.
# To avoid code duplication, the following two functions are defined.
def compute_tp_tensor(state, sos_expr, n_f, perm_pairs, gauge_origin, conv_tol=1e-4):
    tensor = evaluate_property_isr(
        state,  # ExcitedStates object returned by adcc calculation
        sos_expr,  # first term of symbolic SOS expression
        [n],  # indices of summation
        excluded_states=O,  # states excluded from summation (here: ground state)
        excited_state=n_f,  # excited state of interest (here: final state)
        freqs_in=[(w_a, w_f / 2), (w_b, w_f / 2)],  # incoming frequencies
        perm_pairs=perm_pairs,  # pairs to be permuted
        gauge_origin=gauge_origin,  # gauge origin for operator integrals
        conv_tol=conv_tol  # convergence tolerance for response solver
    )
    return tensor

# b1, b2, and b3 are defined by the experimental setup.
def compute_rotatory_strength(tpcd_data, final_state, b1=6, b2=2, b3=-2):
    external_energy = tpcd_data["excitation_energy_uncorrected"][final_state]/2
    Spp = tpcd_data[f"state_{final_state}"]["Spp"]
    Qpp = tpcd_data[f"state_{final_state}"]["Qpp"]
    Mp = tpcd_data[f"state_{final_state}"]["Mp"]
    B1_p = 1/(external_energy**3) * np.einsum("ps,ps->", np.conjugate(Mp), Spp)
    B2_p = 1/(2 * (external_energy**3)) * np.einsum("ps,ps->", np.conjugate(Qpp), Spp)
    B3_p = 1/(external_energy**3) * np.einsum("ss->", np.conjugate(Mp)) * np.einsum("pp->", Spp)
    R_TP_p = 1.0 * b1 * B1_p + b2 * B2_p + 1.0 * b3 * B3_p
    return R_TP_p

# run SCF in PySCF
mol = gto.M(
    atom="""
    O       0.000000     0.000000     0.000000
    O       1.480000     0.000000     0.000000
    H      -0.316648     0.000000     0.895675
    H       1.796648     0.775678    -0.447838
    """,
    unit="Angstrom",
    basis="sto-3g",
)
scfres = scf.RHF(mol)
scfres.kernel()

# run ADC calculation using adcc
state = adcc.run_adc(scfres, method="adc2", n_singlets=2)
print(state.describe())

# define first SOS term
Mp_sos_expr = TransitionMoment(f, m_b, n, shifted=True) \
    * TransitionMoment(n, mup_a, O) / (w_n - w_a)
Spp_sos_expr = TransitionMoment(f, mup_b, n, shifted=True) \
    * TransitionMoment(n, mup_a, O) / (w_n - w_a)
Qpp_sos_expr = TransitionMoment(f, mup_c, n, shifted=True) \
    * TransitionMoment(n, qp_ab, O) / (w_n - w_a)

# define operator-frequency pairs to be permuted
Mp_perm_pairs = [(m_b, w_b), (mup_a, w_a)]
Spp_perm_pairs = [(mup_b, w_b), (mup_a, w_a)]
Qpp_perm_pairs = [(mup_c, w_b), (qp_ab, w_a)]

tensors = {
    "Mp": (Mp_sos_expr, Mp_perm_pairs),
    "Spp": (Spp_sos_expr, Spp_perm_pairs),
    "Qpp": (Qpp_sos_expr, Qpp_perm_pairs),
}

# define gauge origin for gauge origin dependent operator integrals
gauge_origin = "mass_center"
tpcd_data = {}

# compute two-photon tensors 
for n_f in range(2):
    data_state = {}
    for tensor_name, (sos_expr, perm_pairs) in tensors.items():
        tensor = compute_tp_tensor(state, sos_expr, n_f, perm_pairs, gauge_origin)
        if len(tensor.shape) == 3:
            data_state[tensor_name] = np.einsum("bcd,acd->ab", epsilon, tensor)
        else:
            data_state[tensor_name] = tensor

    tpcd_data[f"state_{n_f}"] = data_state

tpcd_data["excitation_energy_uncorrected"] = state.excitation_energy_uncorrected

for ex_state in range(2):
    R_TP_p = compute_rotatory_strength(tpcd_data, ex_state)
    print("The two-photon rotatory strength in velocity gauge for excited state "
          f"{ex_state} is {R_TP_p:.2f} (a.u.).")
    