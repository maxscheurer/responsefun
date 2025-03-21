import adcc
import numpy as np
import pytest
from adcc.Excitation import Excitation
from adcc.misc import assert_allclose_signfix
from respondo.polarizability import (
    complex_polarizability,
    real_polarizability,
    static_polarizability,
)
from respondo.rixs import rixs
from respondo.tpa import tpa_resonant

from responsefun.evaluate_property import (
    evaluate_property_isr,
    evaluate_property_sos,
    evaluate_property_sos_fast,
)
from responsefun.misc import ev2au
from responsefun.SumOverStates import TransitionMoment
from responsefun.symbols_and_labels import (
    O,
    f,
    gamma,
    k,
    m,
    n,
    mu_a,
    mu_b,
    mu_c,
    mu_d,
    p,
    w,
    w_prime,
    w_1,
    w_2,
    w_3,
    w_f,
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


case_list = [(c,) for c in cache.cases]
SOS_expressions = {
    "alpha": (
        (
            TransitionMoment(O, mu_a, n) * TransitionMoment(n, mu_b, O) / (w_n - w)
            + TransitionMoment(O, mu_b, n) * TransitionMoment(n, mu_a, O) / (w_n + w)
        ),
        None,
    ),
    "alpha_complex": (
        (
            TransitionMoment(O, mu_a, n) * TransitionMoment(n, mu_b, O) / (w_n - w - 1j * gamma)
            + TransitionMoment(O, mu_b, n) * TransitionMoment(n, mu_a, O) / (w_n + w + 1j * gamma)
        ),
        None,
    ),
    "rixs_short": (
        (TransitionMoment(f, mu_a, n) * TransitionMoment(n, mu_b, O) / (w_n - w - 1j * gamma)),
        None,
    ),
    "rixs": (
        (
            (TransitionMoment(f, mu_a, n) * TransitionMoment(n, mu_b, O)
            / (w_n - w - 1j * gamma))
            + (TransitionMoment(f, mu_b, n) * TransitionMoment(n, mu_a, O)
            / (w_n + w - w_f + 1j * gamma))
        ),
        None,
    ),
    "tpa_resonant": (
        (
            TransitionMoment(O, mu_a, n) * TransitionMoment(n, mu_b, f) / (w_n - (w_f / 2))
            + TransitionMoment(O, mu_b, n) * TransitionMoment(n, mu_a, f) / (w_n - (w_f / 2))
        ),
        None,
    ),
    "beta": (
        (
            TransitionMoment(O, mu_a, n)
            * TransitionMoment(n, mu_b, k)
            * TransitionMoment(k, mu_c, O)
            / ((w_n - w_o) * (w_k - w_2))
        ),
        [(mu_a, -w_o), (mu_b, w_1), (mu_c, w_2)],
    ),
    "beta_complex": (
        (
            TransitionMoment(O, mu_a, n)
            * TransitionMoment(n, mu_b, k, shifted=True)
            * TransitionMoment(k, mu_c, O)
            / ((w_n - w_o - 1j * gamma) * (w_k - w_2 - 1j * gamma))
        ),
        [(mu_a, -w_o - 1j * gamma), (mu_b, w_1 + 1j * gamma), (mu_c, w_2 + 1j * gamma)],
    ),
    "gamma": (
        (
            TransitionMoment(O, mu_a, n)
            * TransitionMoment(n, mu_b, m)
            * TransitionMoment(m, mu_c, p)
            * TransitionMoment(p, mu_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
        ),
        [(mu_a, -w_o), (mu_b, w_1), (mu_c, w_2), (mu_d, w_3)],
    ),
    "gamma_extra_terms_1": (
        (
            TransitionMoment(O, mu_a, n)
            * TransitionMoment(n, mu_b, O)
            * TransitionMoment(O, mu_c, m)
            * TransitionMoment(m, mu_d, O)
            / ((w_n - w_o) * (w_m - w_3) * (w_m + w_2))
        ),
        [(mu_a, -w_o), (mu_b, w_1), (mu_c, w_2), (mu_d, w_3)],
    ),
    "gamma_extra_terms_2": (
        (
            TransitionMoment(O, mu_a, n)
            * TransitionMoment(n, mu_b, O)
            * TransitionMoment(O, mu_c, m)
            * TransitionMoment(m, mu_d, O)
            / ((w_n - w_o) * (-w_2 - w_3) * (w_m - w_3))
        ),
        [(mu_a, -w_o), (mu_b, w_1), (mu_c, w_2), (mu_d, w_3)],
    ),
}

# TODO: add mcd test as soon as gator-program/respondo#15 is merged
@pytest.mark.parametrize("case", cache.cases)
class TestIsrAgainstRespondo:
    def test_static_polarizability(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        alpha_ref = static_polarizability(refstate, method=method)

        state = adcc.run_adc(refstate, method=method, n_singlets=5)
        alpha_expr = SOS_expressions["alpha"][0]
        freq = (w, 0.0)
        alpha = evaluate_property_isr(state, alpha_expr, [n], freqs_in=freq,
                                      freqs_out=freq, symmetric=True)
        np.testing.assert_allclose(alpha, alpha_ref, atol=1e-7)

    def test_real_polarizability(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        omega = 0.05
        alpha_ref = real_polarizability(refstate, method=method, omega=omega)

        state = adcc.run_adc(refstate, method=method, n_singlets=5)
        alpha_expr = SOS_expressions["alpha_complex"][0]
        freq = (w, omega)
        alpha = evaluate_property_isr(state, alpha_expr, [n], freqs_in=freq,
                                      freqs_out=freq, symmetric=True)
        np.testing.assert_allclose(alpha, alpha_ref, atol=1e-7)

    def test_complex_polarizability(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        omega = 0.05
        gamma_val = ev2au(0.124)
        alpha_ref = complex_polarizability(refstate, method=method, omega=omega, gamma=gamma_val)

        state = adcc.run_adc(refstate, method=method, n_singlets=5)
        alpha_expr = SOS_expressions["alpha_complex"][0]
        freq = (w, omega)
        alpha = evaluate_property_isr(state, alpha_expr, [n], freqs_in=freq, freqs_out=freq,
                                      damping=gamma_val, symmetric=True)
        np.testing.assert_allclose(alpha, alpha_ref, atol=1e-7)

    def test_rixs_short(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        state = adcc.run_adc(refstate, method=method, n_singlets=5)
        omega = 0.05
        freqs_in = (w, omega)
        freqs_out=(w_prime, w-w_f)
        gamma_val = ev2au(0.124)
        rixs_expr = SOS_expressions["rixs_short"][0]

        for ee in state.excitations:
            final_state = ee.index
            excited_state = Excitation(state, final_state, method)

            rixs_ref = rixs(excited_state, omega, gamma_val)
            rixs_short = evaluate_property_isr(
                state, rixs_expr, [n], freqs_in=freqs_in, freqs_out=freqs_out,
                damping=gamma_val, excited_state=final_state
            )
            np.testing.assert_allclose(
                rixs_short, rixs_ref[1], atol=1e-7, err_msg="final_state = {}".format(final_state)
            )

    def test_rixs(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        state = adcc.run_adc(refstate, method=method, n_singlets=5)
        omega = 0.05
        freqs_in = (w, omega)
        freqs_out=(w_prime, w-w_f)
        gamma_val = ev2au(0.124)
        rixs_expr = SOS_expressions["rixs"][0]

        for ee in state.excitations:
            final_state = ee.index
            excited_state = Excitation(state, final_state, method)

            rixs_ref = rixs(excited_state, omega, gamma_val, rotating_wave=False)
            rixs_tens = evaluate_property_isr(
                state, rixs_expr, [n], freqs_in=freqs_in, freqs_out=freqs_out,
                damping=gamma_val, excited_state=final_state
            )
            np.testing.assert_allclose(
                rixs_tens, rixs_ref[1], atol=1e-7, err_msg="final_state = {}".format(final_state)
            )

    def test_tpa_resonant(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        state = adcc.run_adc(refstate, method=method, n_singlets=5)
        tpa_expr = SOS_expressions["tpa_resonant"][0]

        for ee in state.excitations:
            final_state = ee.index
            excited_state = Excitation(state, final_state, method)

            tpa_ref = tpa_resonant(excited_state)
            tpa = evaluate_property_isr(
                state, tpa_expr, [n], excited_state=final_state, freqs_in=[(w_f, w_f)]
            )
            np.testing.assert_allclose(
                tpa, tpa_ref[1], atol=1e-7, err_msg="final_state = {}".format(final_state)
            )


@pytest.mark.parametrize("case", [case for case in cache.cases if case in cache.data_fulldiag])
class TestIsrAgainstSos:
    def test_polarizability(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        alpha_expr = SOS_expressions["alpha_complex"][0]
        gamma_val = ev2au(0.124)
        # static, real and complex polarizability
        value_list = [((w, 0.0), 0.0), ((w, 0.05), 0.0), ((w, 0.03), gamma_val)]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        for tup in value_list:
            alpha_sos = evaluate_property_sos(
                mock_state, alpha_expr, [n], freqs_in=tup[0], freqs_out=tup[0],
                damping=tup[1], symmetric=True
            )
            alpha_isr = evaluate_property_isr(
                state, alpha_expr, [n], freqs_in=tup[0], freqs_out=tup[0],
                damping=tup[1], symmetric=True
            )
            np.testing.assert_allclose(
                alpha_isr,
                alpha_sos,
                atol=1e-7,
                err_msg="w = {}, gamma = {}".format(tup[0][1], tup[1]),
            )

    def test_rixs_short(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        rixs_expr = SOS_expressions["rixs_short"][0]
        gamma_val = ev2au(0.124)
        value_list = [((w, 0.0), 0.0), ((w, 0.05), 0.0), ((w, 1), 0), ((w, 0.03), gamma_val)]
        freqs_out = (w_prime, w-w_f)
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        for tup in value_list:
            for ee in state.excitations:
                final_state = ee.index
                if tup[0][1] == 0.0 and tup[1] == 0.0:
                    with pytest.raises(ZeroDivisionError):
                        evaluate_property_sos(
                            mock_state, rixs_expr, [n], freqs_in=tup[0], freqs_out=freqs_out,
                            damping=tup[1], excited_state=final_state
                        )
                    with pytest.raises(ZeroDivisionError):
                        evaluate_property_isr(
                            state, rixs_expr, [n], freqs_in=tup[0], freqs_out=freqs_out,
                            damping=tup[1], excited_state=final_state
                        )
                else:
                    rixs_sos = evaluate_property_sos(
                        mock_state, rixs_expr, [n], freqs_in=tup[0], freqs_out=freqs_out,
                        damping=tup[1], excited_state=final_state
                    )
                    rixs_isr = evaluate_property_isr(
                        state, rixs_expr, [n], freqs_in=tup[0], freqs_out=freqs_out,
                        damping=tup[1], excited_state=final_state
                    )
                    err_msg = "w = {}, gamma = {}, final_state = {}".format(
                        tup[0][1], tup[1], final_state
                    )
                    assert_allclose_signfix(rixs_isr, rixs_sos, atol=1e-7, err_msg=err_msg)

    def test_rixs(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        rixs_expr = SOS_expressions["rixs"][0]
        freqs_in = (w, 0.05)
        freqs_out = (w_prime, w-w_f)
        gamma_val = ev2au(0.124)
        final_state = 2
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        rixs_sos = evaluate_property_sos(
            mock_state, rixs_expr, [n], freqs_in=freqs_in, freqs_out=freqs_out,
            damping=gamma_val, excited_state=final_state
        )
        rixs_isr = evaluate_property_isr(
            state, rixs_expr, [n], freqs_in=freqs_in, freqs_out=freqs_out,
            damping=gamma_val, excited_state=final_state
        )
        assert_allclose_signfix(rixs_isr, rixs_sos, atol=1e-7)

    def test_tpa_resonant(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        tpa_expr = SOS_expressions["tpa_resonant"][0]
        final_state = 2
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        tpa_sos = evaluate_property_sos(
            mock_state, tpa_expr, [n], excited_state=final_state, freqs_in=[(w_f, w_f)]
        )
        tpa_isr = evaluate_property_isr(
            state, tpa_expr, [n], excited_state=final_state, freqs_in=[(w_f, w_f)]
        )

        assert_allclose_signfix(tpa_isr, tpa_sos, atol=1e-7)

    # def test_first_hyperpolarizability(self, case):
    #     molecule, basis, method = case.split("_")
    #     scfres = run_scf(molecule, basis)
    #     refstate = adcc.ReferenceState(scfres)
    #     beta_expr = SOS_expressions["beta"][0]
    #     perm_pairs = SOS_expressions["beta"][1]
    #     omega_list = [
    #         [(w_o, w_1 + w_2), (w_1, 0.0), (w_2, 0.0)],
    #         [(w_o, w_1 + w_2), (w_1, 0.05), (w_2, 0.05)],
    #         [(w_o, w_1 + w_2), (w_1, -0.05), (w_2, 0.05)],
    #         [(w_o, w_1 + w_2), (w_1, 0.04), (w_2, 0.06)],
    #     ]
    #     mock_state = cache.data_fulldiag[case]
    #     state = adcc.run_adc(refstate, method=method, n_singlets=5)

    #     for omegas in omega_list:
    #         beta_sos = evaluate_property_sos(
    #             mock_state, beta_expr, [n, k], omegas, perm_pairs=perm_pairs
    #         )
    #         beta_isr = evaluate_property_isr(
    #             state, beta_expr, [n, k], omegas, perm_pairs=perm_pairs
    #         )
    #         np.testing.assert_allclose(
    #             beta_isr,
    #             beta_sos,
    #             atol=1e-7,
    #             err_msg="w = {}, gamma = {}".format(tup[0][1], tup[1]),
    #         )


@pytest.mark.parametrize("case", [case for case in cache.cases if case in cache.data_fulldiag])
class TestIsrAgainstSosFast:
    def test_polarizability(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        alpha_expr = SOS_expressions["alpha_complex"][0]
        gamma_val = ev2au(0.124)
        # static, real and complex polarizability
        value_list = [((w, 0.0), 0.0), ((w, 0.05), 0.0), ((w, 0.03), gamma_val)]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        for tup in value_list:
            alpha_sos = evaluate_property_sos_fast(
                mock_state, alpha_expr, [n], freqs_in=tup[0], freqs_out=tup[0], damping=tup[1]
            )
            alpha_isr = evaluate_property_isr(
                state, alpha_expr, [n], freqs_in=tup[0], freqs_out=tup[0],
                damping=tup[1], symmetric=True
            )
            np.testing.assert_allclose(
                alpha_isr,
                alpha_sos,
                atol=1e-7,
                err_msg="w = {}, gamma = {}".format(tup[0][1], tup[1]),
            )

    def test_rixs_short(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        rixs_expr = SOS_expressions["rixs_short"][0]
        gamma_val = ev2au(0.124)
        value_list = [((w, 0.0), 0.0), ((w, 0.05), 0.0), ((w, 1), 0), ((w, 0.03), gamma_val)]
        freqs_out = (w_prime, w-w_f)
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        for tup in value_list:
            for ee in state.excitations:
                final_state = ee.index
                if tup[0][1] == 0.0 and tup[1] == 0.0:
                    with pytest.raises(ZeroDivisionError):
                        evaluate_property_sos_fast(
                            mock_state, rixs_expr, [n], freqs_in=tup[0], freqs_out=freqs_out,
                            damping=tup[1], excited_state=final_state
                        )
                    with pytest.raises(ZeroDivisionError):
                        evaluate_property_isr(
                            state, rixs_expr, [n], freqs_in=tup[0], freqs_out=freqs_out,
                            damping=tup[1], excited_state=final_state
                        )
                else:
                    rixs_sos = evaluate_property_sos_fast(
                        mock_state, rixs_expr, [n], freqs_in=tup[0], freqs_out=freqs_out,
                        damping=tup[1], excited_state=final_state
                    )
                    rixs_isr = evaluate_property_isr(
                        state, rixs_expr, [n], freqs_in=tup[0], freqs_out=freqs_out,
                        damping=tup[1], excited_state=final_state
                    )
                    err_msg = "w = {}, gamma = {}, final_state = {}".format(
                        tup[0][1], tup[1], final_state
                    )
                    assert_allclose_signfix(rixs_isr, rixs_sos, atol=1e-7, err_msg=err_msg)

    def test_rixs(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        rixs_expr = SOS_expressions["rixs"][0]
        freqs_in = (w, 0.05)
        freqs_out = (w_prime, w-w_f)
        gamma_val = ev2au(0.124)
        final_state = 2
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        rixs_sos = evaluate_property_sos_fast(
            mock_state, rixs_expr, [n], freqs_in=freqs_in, freqs_out=freqs_out,
            damping=gamma_val, excited_state=final_state
        )
        rixs_isr = evaluate_property_isr(
            state, rixs_expr, [n], freqs_in=freqs_in, freqs_out=freqs_out,
            damping=gamma_val, excited_state=final_state
        )
        assert_allclose_signfix(rixs_isr, rixs_sos, atol=1e-7)

        # give two different values for the same frequency
        # self.assertRaises(
        #         ValueError, evaluate_property_sos_fast,
        #         mock_state, rixs_expr, [n], [(w, 0.05), (w, 0.03)], gamma_val,
        #         final_state=(f, final_state)
        # )
        # self.assertRaises(
        #         ValueError, evaluate_property_isr,
        #         state, rixs_expr, [n], [(w, 0.05), (w, 0.03)], gamma_val,
        #         final_state=(f, final_state)
        # )

    def test_tpa_resonant(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        tpa_expr = SOS_expressions["tpa_resonant"][0]
        final_state = 2
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        tpa_sos = evaluate_property_sos_fast(
            mock_state, tpa_expr, [n], excited_state=final_state, freqs_in=[(w_f, w_f)]
        )
        tpa_isr = evaluate_property_isr(
            state, tpa_expr, [n], excited_state=final_state, freqs_in=[(w_f, w_f)]
        )

        assert_allclose_signfix(tpa_isr, tpa_sos, atol=1e-7)

        # specify frequency that is not included in the SOS expression
        # self.assertRaises(
        #         ValueError, evaluate_property_sos_fast,
        #         mock_state, tpa_expr, [n], (w, 0.05), final_state=(f, final_state)
        # )
        # self.assertRaises(
        #         ValueError, evaluate_property_isr,
        #         state, tpa_expr, [n], (w, 0.05), final_state=(f, final_state)
        # )

    def test_first_hyperpolarizability(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        beta_expr, perm_pairs = SOS_expressions["beta"]
        omega_list = [
            [(w_o, w_1 + w_2), (w_1, 0.0), (w_2, 0.0)],
            [(w_o, w_1 + w_2), (w_1, 0.05), (w_2, 0.05)],
            [(w_o, w_1 + w_2), (w_1, -0.05), (w_2, 0.05)],
            [(w_o, w_1 + w_2), (w_1, 0.04), (w_2, 0.06)],
        ]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        for omegas in omega_list:
            beta_sos = evaluate_property_sos_fast(
                mock_state, beta_expr, [n, k], freqs_in=omegas[1:],
                freqs_out=omegas[0], perm_pairs=perm_pairs
            )
            beta_isr = evaluate_property_isr(
                state, beta_expr, [n, k], freqs_in=omegas[1:],
                freqs_out=omegas[0], perm_pairs=perm_pairs
            )
            np.testing.assert_allclose(beta_isr, beta_sos, atol=1e-7)

        # give wrong indices of summation
        with pytest.raises(ValueError):
            evaluate_property_sos_fast(
                mock_state, beta_expr, [n, p], freqs_in=omega_list[0][1:],
                freqs_out=omega_list[0][0], perm_pairs=perm_pairs
            )
        with pytest.raises(ValueError):
            evaluate_property_isr(
                state, beta_expr, [n, p], freqs_in=omega_list[0][1:],
                freqs_out=omega_list[0][0], perm_pairs=perm_pairs
            )

    def test_complex_first_hyperpolarizability(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        beta_expr, perm_pairs = SOS_expressions["beta_complex"]
        omega_list = [
            [(w_o, w_1 + w_2), (w_1, 0.05), (w_2, 0.05)],
            [(w_o, w_1 + w_2), (w_1, -0.05), (w_2, 0.05)],
            [(w_o, w_1 + w_2), (w_1, 0.04), (w_2, 0.06)],
        ]
        gamma_val = ev2au(0.124)
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        for omegas in omega_list:
            beta_sos = evaluate_property_sos_fast(
                mock_state, beta_expr, [n, k], freqs_in=omegas[1:], freqs_out=omegas[0],
                damping=gamma_val, perm_pairs=perm_pairs, excluded_states=O
            )
            beta_isr = evaluate_property_isr(
                state, beta_expr, [n, k], freqs_in=omegas[1:], freqs_out=omegas[0],
                damping=gamma_val, perm_pairs=perm_pairs, excluded_states=O
            )
            np.testing.assert_allclose(beta_isr, beta_sos, atol=1e-7)

    @pytest.mark.slow
    def test_second_hyperpolarizability(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        gamma_expr, perm_pairs = SOS_expressions["gamma"]
        gamma_expr_extra, perm_pairs_extra = SOS_expressions["gamma_extra_terms_1"]
        omega_list = [
            [(w_o, w_1 + w_2 + w_3), (w_1, 0.0), (w_2, 0.0), (w_3, 0.0)],
            [(w_o, w_1 + w_2 + w_3), (w_1, 0.0), (w_2, 0.0), (w_3, 0.05)],
            [(w_o, w_1 + w_2 + w_3), (w_1, 0.04), (w_2, 0.05), (w_3, 0.06)],
        ]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)
        for omegas in omega_list:
            gamma_sos = evaluate_property_sos_fast(
                mock_state, gamma_expr, [n, m, p], freqs_in=omegas[1:], freqs_out=omegas[0],
                perm_pairs=perm_pairs, extra_terms=False
            )
            gamma_isr = evaluate_property_isr(
                state, gamma_expr, [n, m, p], freqs_in=omegas[1:], freqs_out=omegas[0],
                perm_pairs=perm_pairs, extra_terms=False
            )
            gamma_sos_extra = evaluate_property_sos_fast(
                mock_state, gamma_expr_extra, [n, m], freqs_in=omegas[1:], freqs_out=omegas[0],
                perm_pairs=perm_pairs_extra, extra_terms=False
            )
            gamma_isr_extra = evaluate_property_isr(
                state, gamma_expr_extra, [n, m], freqs_in=omegas[1:], freqs_out=omegas[0],
                perm_pairs=perm_pairs_extra, extra_terms=False
            )
            gamma_isr_tot = gamma_isr - gamma_isr_extra
            gamma_sos_tot = gamma_sos - gamma_sos_extra
            np.testing.assert_allclose(gamma_isr_tot, gamma_sos_tot, atol=1e-7)

    @pytest.mark.slow
    def test_extra_terms_second_hyperpolarizability(self, case):
        molecule, basis, method = case.split("_")
        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        gamma_expr, perm_pairs = SOS_expressions["gamma_extra_terms_2"]
        omega_list = [
            [(w_o, w_1 + w_2 + w_3), (w_1, 0.0), (w_2, 0.0), (w_3, 0.0)],
            [(w_o, w_1 + w_2 + w_3), (w_1, 0.0), (w_2, 0.0), (w_3, 0.05)],
            [(w_o, w_1 + w_2 + w_3), (w_1, 0.04), (w_2, 0.05), (w_3, 0.06)],
        ]
        mock_state = cache.data_fulldiag[case]
        state = adcc.run_adc(refstate, method=method, n_singlets=5)

        for omegas in omega_list:
            if omegas[1][1] == 0.0 and omegas[2][1] == 0.0:
                with pytest.raises(ZeroDivisionError):
                    evaluate_property_sos_fast(
                        mock_state, gamma_expr, [n, m], freqs_in=omegas[1:], freqs_out=omegas[0],
                        perm_pairs=perm_pairs, extra_terms=False
                    )
                with pytest.raises(ZeroDivisionError):
                    evaluate_property_isr(
                        state, gamma_expr, [n, m], freqs_in=omegas[1:], freqs_out=omegas[0],
                        perm_pairs=perm_pairs, extra_terms=False
                    )
            else:
                gamma_sos = evaluate_property_sos_fast(
                    mock_state, gamma_expr, [n, m], freqs_in=omegas[1:], freqs_out=omegas[0],
                        perm_pairs=perm_pairs, extra_terms=False
                )
                gamma_isr = evaluate_property_isr(
                    state, gamma_expr, [n, m], freqs_in=omegas[1:], freqs_out=omegas[0],
                        perm_pairs=perm_pairs, extra_terms=False
                )
                np.testing.assert_allclose(gamma_isr, gamma_sos, atol=1e-7)
