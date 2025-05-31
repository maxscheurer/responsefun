import adcc
import numpy as np
import pytest

from responsefun.evaluate_property import (
    evaluate_property_isr,
    evaluate_property_sos_fast,
)
from responsefun.SumOverStates import TransitionMoment
from responsefun.symbols_and_labels import (
    O,
    k,
    mu_a,
    mu_b,
    mu_c,
    n,
    q_ab,
    q_bc,
    q_cd,
    q_de,
    w,
    w_1,
    w_2,
    w_k,
    w_n,
    w_o,
)
from responsefun.testdata import cache
from responsefun.testdata.static_data import xyz


def run_scf(molecule, basis, backend="pyscf"):
    scfres = adcc.backends.run_hf(
        backend,
        xyz=xyz[molecule],
        basis=basis,
    )
    return scfres


SOS_alpha_like = {
    "ab": (
        TransitionMoment(O, q_ab, n) * TransitionMoment(n, mu_c, O) / (w_n - w)
        + TransitionMoment(O, mu_c, n) * TransitionMoment(n, q_ab, O) / (w_n + w)
    ),
    "bc": (
        TransitionMoment(O, mu_a, n) * TransitionMoment(n, q_bc, O) / (w_n - w)
        + TransitionMoment(O, q_bc, n) * TransitionMoment(n, mu_a, O) / (w_n + w)
    ),
    "abcd": (
        TransitionMoment(O, q_ab, n) * TransitionMoment(n, q_cd, O) / (w_n - w)
        + TransitionMoment(O, q_cd, n) * TransitionMoment(n, q_ab, O) / (w_n + w)
    ),
}


SOS_beta_like = {
    "ab": (
        TransitionMoment(O, q_ab, n)
        * TransitionMoment(n, mu_b, k)
        * TransitionMoment(k, mu_c, O)
        / ((w_n - w_o) * (w_k - w_2))
    ),
    "cd": (
        TransitionMoment(O, mu_a, n)
        * TransitionMoment(n, mu_b, k)
        * TransitionMoment(k, q_cd, O)
        / ((w_n - w_o) * (w_k - w_2))
    ),
    "abde": (
        TransitionMoment(O, q_ab, n)
        * TransitionMoment(n, mu_c, k)
        * TransitionMoment(k, q_de, O)
        / ((w_n - w_o) * (w_k - w_2))
    ),
}

@pytest.mark.parametrize("ops", SOS_alpha_like.keys())
class TestAlphaLike:
    def test_h2o_sto3g_adc2(self, ops):
        case = "h2o_sto3g_adc2"
        if case not in cache.data_fulldiag:
            pytest.skip(f"{case} cache file not available.")
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        expr = SOS_alpha_like[ops]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)
        freq = (w, 0.5)
        alpha_sos = evaluate_property_sos_fast(mock_state, expr, [n], freqs_in=freq, freqs_out=freq)
        alpha_isr = evaluate_property_isr(state, expr, [n], freqs_in=freq, freqs_out=freq)
        np.testing.assert_allclose(alpha_isr, alpha_sos, atol=1e-8)


@pytest.mark.parametrize("ops", SOS_beta_like.keys())
class TestBetaLike:
    def test_h2o_sto3g_adc2(self, ops):
        case = "h2o_sto3g_adc2"
        if case not in cache.data_fulldiag:
            pytest.skip(f"{case} cache file not available.")
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        expr = SOS_beta_like[ops]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        freqs_in = [(w_1, 0.5), (w_2, 0.5)]
        freqs_out = (w_o, 1)
        beta_sos = evaluate_property_sos_fast(mock_state, expr, [n, k], freqs_in=freqs_in,
                                              freqs_out=freqs_out, extra_terms=False)
        beta_isr = evaluate_property_isr(state, expr, [n, k], freqs_in=freqs_in,
                                         freqs_out=freqs_out, extra_terms=False)
        np.testing.assert_allclose(beta_isr, beta_sos, atol=1e-8)