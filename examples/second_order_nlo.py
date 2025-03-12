"""
Compute second-order nonlinear optical properties (see 10.1021/acs.jctc.3c00456)
with an SOS expression for the first-order hyperpolarizability according to
Eq. (5.187) in 10.1002/9781118794821.
"""
import adcc
from pyscf import gto, scf
import numpy as np

from responsefun.evaluate_property import evaluate_property_isr
from responsefun.SumOverStates import TransitionMoment
from responsefun.symbols_and_labels import (
    O,
    n,
    op_a,
    op_b,
    op_c,
    p,
    w_1,
    w_2,
    w_n,
    w_o,
    w_p,
)


def compute_beta_parallel(beta_tens, dip_mom):
    beta_parallel = (
        (1/(5*np.linalg.norm(dip_mom)))
        * (
            np.einsum("i,ijj->", dip_mom, beta_tens)
            + np.einsum("i,jij->", dip_mom, beta_tens)
            + np.einsum("i,jji->", dip_mom, beta_tens)
        )
    )
    return beta_parallel


# run SCF in PySCF
mol = gto.M(
    atom="""
    O        0.000000    0.000000    0.115082
    H        0.000000    0.767545   -0.460329
    H        0.000000   -0.767545   -0.460329
    """,
    unit="Angstrom",
    basis="aug-cc-pvdz",
)
scfres = scf.RHF(mol)
scfres.kernel()

w_ruby = 0.0656

# run ADC(2) calculation using adcc
state = adcc.adc2(scfres, n_singlets=5)
dip_mom = state.ground_state.dipole_moment(state.method.level)

# compute the first hyperpolarizability tensor
beta_term = (
    TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, p, shifted=True)
    * TransitionMoment(p, op_c, O) / ((w_n - w_o) * (w_p - w_2))
)
processes = {
    "static": (0.0, 0.0), "OR": (w_ruby, -w_ruby),
    "EOPE": (w_ruby, 0.0), "SHG": (w_ruby, w_ruby)
}
for process, freqs in processes.items():
    # the minus sign is needed, because the negative charge is not yet included
    # in the operator definitions
    # TODO: remove minus after adc-connect/adcc#190 is merged
    beta_tens = -1.0 * evaluate_property_isr(
        state, beta_term, [n, p],
        perm_pairs=[(op_a, -w_o), (op_b, w_1), (op_c, w_2)],
        freqs_in=[(w_1, freqs[0]), (w_2, freqs[1])],
        freqs_out=(w_o, w_1+w_2),
        excluded_states=O,
        conv_tol=1e-5,
    )
    print(process)
    print(f"beta_parallel = {compute_beta_parallel(beta_tens, dip_mom):.2f} a.u.")