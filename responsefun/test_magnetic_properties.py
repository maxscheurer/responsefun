import unittest
import adcc
import numpy as np
from scipy.constants import physical_constants

from responsefun.testdata.static_data import xyz
from responsefun.testdata import cache
from responsefun.symbols_and_labels import *
from responsefun.SumOverStates import TransitionMoment, SumOverStates
from responsefun.evaluate_property import evaluate_property_isr, evaluate_property_sos, evaluate_property_sos_fast
from adcc.misc import expand_test_templates, assert_allclose_signfix


Hartree = physical_constants["hartree-electron volt relationship"][0]


def run_scf(molecule, basis, backend="pyscf"):
    scfres = adcc.backends.run_hf(
        backend, xyz=xyz[molecule],
        basis=basis,
    )
    return scfres


SOS_alpha_like = {
        "a":
        (
            TransitionMoment(O, opm_a, n) * TransitionMoment(n, op_b, O) / (w_n - w)
            + TransitionMoment(O, op_b, n) * TransitionMoment(n, opm_a, O) / (w_n + w)
        ),
        "b":
        (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, opm_b, O) / (w_n - w)
            + TransitionMoment(O, opm_b, n) * TransitionMoment(n, op_a, O) / (w_n + w)
        ),
        "ab":
        (
            TransitionMoment(O, opm_a, n) * TransitionMoment(n, opm_b, O) / (w_n - w)
            + TransitionMoment(O, opm_b, n) * TransitionMoment(n, opm_a, O) / (w_n + w)
        )
}
alpha_list = [(c, ) for c in list(SOS_alpha_like.keys())]


SOS_beta_like = {
        "a":
        (
            TransitionMoment(O, opm_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O)
            / ((w_n - w_o) * (w_k - w_2))
        ),
        "b":
        (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, opm_b, k) * TransitionMoment(k, op_c, O)
            / ((w_n - w_o) * (w_k - w_2))
        ),
        "c":
        (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, opm_c, O)
            / ((w_n - w_o) * (w_k - w_2))
        ),
        "ac":
        (
            TransitionMoment(O, opm_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, opm_c, O)
            / ((w_n - w_o) * (w_k - w_2))
        ),
        "bc":
        (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, opm_b, k) * TransitionMoment(k, opm_c, O)
            / ((w_n - w_o) * (w_k - w_2))
        ),
        "abc":
        (
            TransitionMoment(O, opm_a, n) * TransitionMoment(n, opm_b, k) * TransitionMoment(k, opm_c, O)
            / ((w_n - w_o) * (w_k - w_2))
        )
}
beta_list = [(c, ) for c in list(SOS_beta_like.keys())]


SOS_gamma_like = {
        "a":
        (
            TransitionMoment(O, opm_a, n) * TransitionMoment(n, op_b, m)
            * TransitionMoment(m, op_c, p) * TransitionMoment(p, op_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
        ),
        "b":
        (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, opm_b, m)
            * TransitionMoment(m, op_c, p) * TransitionMoment(p, op_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
        ),
        "d":
        (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m)
            * TransitionMoment(m, op_c, p) * TransitionMoment(p, opm_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
        ),
        "ac":
        (
            TransitionMoment(O, opm_a, n) * TransitionMoment(n, op_b, m)
            * TransitionMoment(m, opm_c, p) * TransitionMoment(p, op_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
        ),
        "ad":
        (
            TransitionMoment(O, opm_a, n) * TransitionMoment(n, op_b, m)
            * TransitionMoment(m, op_c, p) * TransitionMoment(p, opm_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
        ),
        "cd":
        (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m)
            * TransitionMoment(m, opm_c, p) * TransitionMoment(p, opm_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
        ),
        "abd":
        (
            TransitionMoment(O, opm_a, n) * TransitionMoment(n, opm_b, m)
            * TransitionMoment(m, op_c, p) * TransitionMoment(p, opm_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
        ),
        "bcd":
        (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, opm_b, m)
            * TransitionMoment(m, opm_c, p) * TransitionMoment(p, opm_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
        ),
        "abcd":
        (
            TransitionMoment(O, opm_a, n) * TransitionMoment(n, opm_b, m)
            * TransitionMoment(m, opm_c, p) * TransitionMoment(p, opm_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
        )
}
gamma_list = [(c, ) for c in list(SOS_gamma_like.keys())]


SOS_delta_like = {
        "a":
        (
            TransitionMoment(O, opm_a, n) * TransitionMoment(n, op_b, m)
            * TransitionMoment(m, op_c, p) * TransitionMoment(p, op_d, k) * TransitionMoment(k, op_e, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3) * (w_k - w_2))
        ),
        "b":
        (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, opm_b, m)
            * TransitionMoment(m, op_c, p) * TransitionMoment(p, op_d, k) * TransitionMoment(k, op_e, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3) * (w_k - w_2))
        ),
        "e":
        (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m)
            * TransitionMoment(m, op_c, p) * TransitionMoment(p, op_d, k) * TransitionMoment(k, opm_e, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3) * (w_k - w_2))
        ),
        "ae":
        (
            TransitionMoment(O, opm_a, n) * TransitionMoment(n, op_b, m)
            * TransitionMoment(m, op_c, p) * TransitionMoment(p, op_d, k) * TransitionMoment(k, opm_e, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3) * (w_k - w_2))
        )
}
delta_list = [(c, ) for c in list(SOS_delta_like.keys())]


@expand_test_templates(alpha_list)
class TestAlphaLike(unittest.TestCase):
    def template_h2o_sto3g_adc2(self, ops):
        case = "h2o_sto3g_adc2"
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        expr = SOS_alpha_like[ops]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        alpha_sos = evaluate_property_sos_fast(mock_state, expr, [n], [(w, 0.5)])
        alpha_isr = evaluate_property_isr(state, expr, [n], [(w, 0.5)])
        np.testing.assert_allclose(alpha_isr, alpha_sos, atol=1e-8)


@expand_test_templates(beta_list)
class TestBetaLike(unittest.TestCase):
    def template_h2o_sto3g_adc2(self, ops):
        case = "h2o_sto3g_adc2"
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        expr = SOS_beta_like[ops]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)
        
        omegas = [(w_o, 1), (w_2, 0.5)]
        beta_sos = evaluate_property_sos_fast(mock_state, expr, [n, k], omegas, extra_terms=False)
        beta_isr = evaluate_property_isr(state, expr, [n, k], omegas, extra_terms=False)
        np.testing.assert_allclose(beta_isr, beta_sos, atol=1e-8)


@expand_test_templates(gamma_list)
class TestGammaLike(unittest.TestCase):
    def template_h2o_sto3g_adc2(self, ops):
        case = "h2o_sto3g_adc2"
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        expr = SOS_gamma_like[ops]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        omegas = [(w_o, 1), (w_2, 0.5), (w_3, 0.5)]
        gamma_sos = evaluate_property_sos_fast(mock_state, expr, [n, m, p], omegas, extra_terms=False)
        gamma_isr = evaluate_property_isr(state, expr, [n, m, p], omegas, extra_terms=False)
        np.testing.assert_allclose(gamma_isr, gamma_sos, atol=1e-8)


@expand_test_templates(delta_list)
class TestDeltaLike(unittest.TestCase):
    def template_h2o_sto3g_adc2(self, ops):
        case = "h2o_sto3g_adc2"
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        expr = SOS_delta_like[ops]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        omegas = [(w_o, 1), (w_2, 0.5), (w_3, 0.3)]
        delta_sos = evaluate_property_sos_fast(mock_state, expr, [n, m, p, k], omegas, extra_terms=False)
        delta_isr = evaluate_property_isr(state, expr, [n, m, p, k], omegas, extra_terms=False)
        np.testing.assert_allclose(delta_isr, delta_sos, atol=1e-8)


class TestCmPara(unittest.TestCase):
    def test_h2o_sto3g_adc2(self):
        case = "h2o_sto3g_adc2"
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        term = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m)
            * TransitionMoment(m, opm_c, p) * TransitionMoment(p, opm_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
        )
        sos = SumOverStates(
                term, [n, m, p], perm_pairs=[(op_a, -w_o), (op_b, w_1), (opm_c, w_2), (opm_d, w_3)]
        )
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        omegas = [(w_o, 0.5), (w_1, 0.5), (w_2, 0.0), (w_3, 0.0)]
        for t in sos.expr.args:
            cm_para_sos = evaluate_property_sos_fast(mock_state, t, [n, m, p], omegas, extra_terms=False)
            cm_para_isr = evaluate_property_isr(state, t, [n, m, p], omegas, extra_terms=False)
            np.testing.assert_allclose(cm_para_isr, cm_para_sos, atol=1e-8, err_msg=f"{t}")
