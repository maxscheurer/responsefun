from pyscf import gto, scf
import adcc
from responsefun.symbols_and_labels import (
    op_a, op_b, op_c, op_d, O, n, m, p, w_n, w_m, w_p, w_1, w_2, w_3, w_o
)
from responsefun.SumOverStates import TransitionMoment
from responsefun.evaluate_property import evaluate_property_isr

# run SCF in PySCF
mol = gto.M(
    atom="""
    8        0.000000    0.000000    0.115082
    1        0.000000    0.767545   -0.460329
    1        0.000000   -0.767545   -0.460329
    """,
    unit="Angstrom",
    basis="aug-cc-pvdz"
)
scfres = scf.RHF(mol)
scfres.kernel()

w_ruby = 0.0656

# run ADC(2) calculation using adcc
state = adcc.adc2(scfres, n_singlets=5)
# compute the second hyperpolarizability tensor
gamma_term_I = (
    TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m)
    * TransitionMoment(m, op_c, p) * TransitionMoment(p, op_d, O)
    / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
)
gamma_term_II = (
    TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O)
    * TransitionMoment(O, op_c, m) * TransitionMoment(m, op_d, O)
    / ((w_n - w_o) * (w_m - w_3) * (w_m + w_2))
)
perm_pairs = [(op_a, -w_o), (op_b, w_1), (op_c, w_2), (op_d, w_3)]
processes = {
    "static": (0.0, 0.0, 0.0), "dcOR": (w_ruby, -w_ruby, 0.0),
    "EOKE": (w_ruby, 0.0, 0.0), "IDRI": (w_ruby, -w_ruby, w_ruby),
    "ESHG": (w_ruby, w_ruby, 0.0), "THG": (w_ruby, w_ruby, w_ruby)
}
for process, freqs in processes.items():
    omegas = [(w_o, w_1+w_2+w_3), (w_1, freqs[0]),
              (w_2, freqs[1]), (w_3, freqs[2])]
    gamma_tens_I = evaluate_property_isr(
        state, gamma_term_I, [n, m, p], omegas=omegas,
        perm_pairs=perm_pairs, extra_terms=False, conv_tol=1e-5
    )
    gamma_tens_II = evaluate_property_isr(
        state, gamma_term_II, [n, m], omegas=omegas,
        perm_pairs=perm_pairs, extra_terms=False, conv_tol=1e-5
    )
    gamma_tens = gamma_tens_I - gamma_tens_II
    print(process)
    print(gamma_tens)
