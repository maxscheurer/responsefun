import adcc
import numpy as np
import pytest

from responsefun.evaluate_property import (
    evaluate_property_isr,
    evaluate_property_sos_fast,
)
from responsefun.SumOverStates import SumOverStates, TransitionMoment
from responsefun.symbols_and_labels import (
    O,
    k,
    m,
    n,
    op_a,
    op_b,
    op_c,
    op_d,
    op_e,
    opm_a,
    opm_b,
    opm_c,
    opm_d,
    opm_e,
    p,
    w,
    w_1,
    w_2,
    w_3,
    w_4,
    w_k,
    w_m,
    w_n,
    w_o,
    w_p,
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
    "a": (
        TransitionMoment(O, opm_a, n) * TransitionMoment(n, op_b, O) / (w_n - w)
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, opm_a, O) / (w_n + w)
    ),
    "b": (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, opm_b, O) / (w_n - w)
        + TransitionMoment(O, opm_b, n) * TransitionMoment(n, op_a, O) / (w_n + w)
    ),
    "ab": (
        TransitionMoment(O, opm_a, n) * TransitionMoment(n, opm_b, O) / (w_n - w)
        + TransitionMoment(O, opm_b, n) * TransitionMoment(n, opm_a, O) / (w_n + w)
    ),
}


SOS_beta_like = {
    "a": (
        TransitionMoment(O, opm_a, n)
        * TransitionMoment(n, op_b, k)
        * TransitionMoment(k, op_c, O)
        / ((w_n - w_o) * (w_k - w_2))
    ),
    "b": (
        TransitionMoment(O, op_a, n)
        * TransitionMoment(n, opm_b, k)
        * TransitionMoment(k, op_c, O)
        / ((w_n - w_o) * (w_k - w_2))
    ),
    "c": (
        TransitionMoment(O, op_a, n)
        * TransitionMoment(n, op_b, k)
        * TransitionMoment(k, opm_c, O)
        / ((w_n - w_o) * (w_k - w_2))
    ),
    "ac": (
        TransitionMoment(O, opm_a, n)
        * TransitionMoment(n, op_b, k)
        * TransitionMoment(k, opm_c, O)
        / ((w_n - w_o) * (w_k - w_2))
    ),
    "bc": (
        TransitionMoment(O, op_a, n)
        * TransitionMoment(n, opm_b, k)
        * TransitionMoment(k, opm_c, O)
        / ((w_n - w_o) * (w_k - w_2))
    ),
    "abc": (
        TransitionMoment(O, opm_a, n)
        * TransitionMoment(n, opm_b, k)
        * TransitionMoment(k, opm_c, O)
        / ((w_n - w_o) * (w_k - w_2))
    ),
}


SOS_gamma_like = {
    "a": (
        TransitionMoment(O, opm_a, n)
        * TransitionMoment(n, op_b, m)
        * TransitionMoment(m, op_c, p)
        * TransitionMoment(p, op_d, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    ),
    "b": (
        TransitionMoment(O, op_a, n)
        * TransitionMoment(n, opm_b, m)
        * TransitionMoment(m, op_c, p)
        * TransitionMoment(p, op_d, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    ),
    "d": (
        TransitionMoment(O, op_a, n)
        * TransitionMoment(n, op_b, m)
        * TransitionMoment(m, op_c, p)
        * TransitionMoment(p, opm_d, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    ),
    "ac": (
        TransitionMoment(O, opm_a, n)
        * TransitionMoment(n, op_b, m)
        * TransitionMoment(m, opm_c, p)
        * TransitionMoment(p, op_d, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    ),
    "ad": (
        TransitionMoment(O, opm_a, n)
        * TransitionMoment(n, op_b, m)
        * TransitionMoment(m, op_c, p)
        * TransitionMoment(p, opm_d, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    ),
    "cd": (
        TransitionMoment(O, op_a, n)
        * TransitionMoment(n, op_b, m)
        * TransitionMoment(m, opm_c, p)
        * TransitionMoment(p, opm_d, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    ),
    "abd": (
        TransitionMoment(O, opm_a, n)
        * TransitionMoment(n, opm_b, m)
        * TransitionMoment(m, op_c, p)
        * TransitionMoment(p, opm_d, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    ),
    "bcd": (
        TransitionMoment(O, op_a, n)
        * TransitionMoment(n, opm_b, m)
        * TransitionMoment(m, opm_c, p)
        * TransitionMoment(p, opm_d, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    ),
    "abcd": (
        TransitionMoment(O, opm_a, n)
        * TransitionMoment(n, opm_b, m)
        * TransitionMoment(m, opm_c, p)
        * TransitionMoment(p, opm_d, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    ),
}


SOS_delta_like = {
    "a": (
        TransitionMoment(O, opm_a, n)
        * TransitionMoment(n, op_b, m)
        * TransitionMoment(m, op_c, p)
        * TransitionMoment(p, op_d, k)
        * TransitionMoment(k, op_e, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3) * (w_k - w_2))
    ),
    "b": (
        TransitionMoment(O, op_a, n)
        * TransitionMoment(n, opm_b, m)
        * TransitionMoment(m, op_c, p)
        * TransitionMoment(p, op_d, k)
        * TransitionMoment(k, op_e, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3) * (w_k - w_2))
    ),
    "e": (
        TransitionMoment(O, op_a, n)
        * TransitionMoment(n, op_b, m)
        * TransitionMoment(m, op_c, p)
        * TransitionMoment(p, op_d, k)
        * TransitionMoment(k, opm_e, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3) * (w_k - w_2))
    ),
    "ae": (
        TransitionMoment(O, opm_a, n)
        * TransitionMoment(n, op_b, m)
        * TransitionMoment(m, op_c, p)
        * TransitionMoment(p, op_d, k)
        * TransitionMoment(k, opm_e, O)
        / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3) * (w_k - w_2))
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


@pytest.mark.slow
@pytest.mark.parametrize("ops", SOS_gamma_like.keys())
class TestGammaLike:
    def test_h2o_sto3g_adc2(self, ops):
        case = "h2o_sto3g_adc2"
        if case not in cache.data_fulldiag:
            pytest.skip(f"{case} cache file not available.")
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        expr = SOS_gamma_like[ops]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        freqs_in = [(w_1, 0.0), (w_2, 0.5), (w_3, 0.5)]
        freqs_out = (w_o, 1)
        gamma_sos = evaluate_property_sos_fast(
            mock_state, expr, [n, m, p], freqs_in=freqs_in,
            freqs_out=freqs_out, extra_terms=False
        )
        gamma_isr = evaluate_property_isr(state, expr, [n, m, p], freqs_in=freqs_in,
                                          freqs_out=freqs_out, extra_terms=False)
        np.testing.assert_allclose(gamma_isr, gamma_sos, atol=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize("ops", SOS_delta_like.keys())
class TestDeltaLike:
    def test_h2o_sto3g_adc2(self, ops):
        case = "h2o_sto3g_adc2"
        if case not in cache.data_fulldiag:
            pytest.skip(f"{case} cache file not available.")
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        expr = SOS_delta_like[ops]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        freqs_in = [(w_1, 0.2), (w_2, 0.5), (w_3, 0.3), (w_4, 0.0)]
        freqs_out = (w_o, 1)
        delta_sos = evaluate_property_sos_fast(
            mock_state, expr, [n, m, p, k], freqs_in=freqs_in,
            freqs_out=freqs_out, extra_terms=False
        )
        delta_isr = evaluate_property_isr(state, expr, [n, m, p, k], freqs_in=freqs_in,
                                          freqs_out=freqs_out, extra_terms=False)
        np.testing.assert_allclose(delta_isr, delta_sos, atol=1e-8)


@pytest.mark.slow
class TestCottonMoutonPara:
    def test_h2o_sto3g_adc2(self):
        case = "h2o_sto3g_adc2"
        if case not in cache.data_fulldiag:
            pytest.skip(f"{case} cache file not available.")
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        term = (
            TransitionMoment(O, op_a, n)
            * TransitionMoment(n, op_b, m)
            * TransitionMoment(m, opm_c, p)
            * TransitionMoment(p, opm_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
        )
        sos = SumOverStates(
            term, [n, m, p], perm_pairs=[(op_a, -w_o), (op_b, w_1), (opm_c, w_2), (opm_d, w_3)]
        )
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        freqs_in = [(w_1, 0.5), (w_2, 0.0), (w_3, 0.0)]
        freqs_out = (w_o, 0.5)
        for t in sos.expr.args:
            cm_para_sos = evaluate_property_sos_fast(
                mock_state, t, [n, m, p], freqs_in=freqs_in,
                freqs_out=freqs_out, extra_terms=False
            )
            cm_para_isr = evaluate_property_isr(state, t, [n, m, p], freqs_in=freqs_in,
                                                freqs_out=freqs_out, extra_terms=False)
            np.testing.assert_allclose(cm_para_isr, cm_para_sos, atol=1e-8, err_msg=f"{t}")