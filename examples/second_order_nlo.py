import adcc
from pyscf import gto, scf

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
# compute the first hyperpolarizability tensor
beta_term = (
    TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, p)
    * TransitionMoment(p, op_c, O) / ((w_n - w_o) * (w_p - w_2))
)
processes = {
    "static": (0.0, 0.0), "OR": (w_ruby, -w_ruby),
    "EOPE": (w_ruby, 0.0), "SHG": (w_ruby, w_ruby)
}
for process, freqs in processes.items():
    beta_tens = evaluate_property_isr(
        state, beta_term, [n, p],
        omegas=[(w_o, w_1+w_2), (w_1, freqs[0]), (w_2, freqs[1])],
        perm_pairs=[(op_a, -w_o), (op_b, w_1), (op_c, w_2)]
    )
    print(process)
    print(beta_tens)
