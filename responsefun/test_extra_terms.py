import adcc
import numpy as np
import pytest
from scipy.constants import physical_constants

from responsefun.evaluate_property import evaluate_property_isr
from responsefun.SumOverStates import TransitionMoment
from responsefun.symbols_and_labels import (
    O,
    k,
    n,
    op_a,
    op_b,
    op_c,
    w_1,
    w_2,
    w_k,
    w_n,
    w_o,
)
from responsefun.testdata import cache
from responsefun.testdata.static_data import xyz

Hartree = physical_constants["hartree-electron volt relationship"][0]


def run_scf(molecule, basis, backend="pyscf"):
    scfres = adcc.backends.run_hf(
        backend,
        xyz=xyz[molecule],
        basis=basis,
    )
    return scfres


case_list = [(c,) for c in cache.cases]
SOS_expressions = {
    "beta1": (
        (
            TransitionMoment(O, op_a, n)
            * TransitionMoment(n, op_b, k)
            * TransitionMoment(k, op_c, O)
            / ((w_n - w_o) * (w_k - w_2))
        ),
        [(op_a, -w_o), (op_b, w_1), (op_c, w_2)],
    ),
    "beta2": (
        (
            TransitionMoment(O, op_a, n)
            * TransitionMoment(n, op_b, k, shifted=True)
            * TransitionMoment(k, op_c, O)
            / ((w_n - w_o) * (w_k - w_2))
        ),
        [(op_a, -w_o), (op_b, w_1), (op_c, w_2)],
    ),    
}


@pytest.mark.parametrize("case", cache.cases)
class TestExtraTerms:
    def test_first_hyperpolarizability(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)        
        state = adcc.run_adc(scfres, method=method, n_singlets=5)

        beta1_expr, perm_pairs = SOS_expressions["beta1"]
        beta2_expr = SOS_expressions["beta2"][0]
        omegas = [(w_o, w_1 + w_2), (w_1, 0.05), (w_2, 0.05)]
        beta1_tens = evaluate_property_isr(
            state, beta1_expr, [n, k], omegas, perm_pairs=perm_pairs
        )
        beta2_tens = evaluate_property_isr(
            state, beta2_expr, [n, k], omegas, perm_pairs=perm_pairs, excluded_states=O
        )
        np.testing.assert_allclose(beta1_tens, beta2_tens, atol=1e-7)