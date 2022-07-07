import numpy as np
from sympy import Symbol, Mul, Add, Pow, Integer, Float, Abs
from sympy import simplify, solveset, S
from responsefun.response_operators import TransitionFrequency


def _fix_equal_to_zero(term, atol=1e-7):
    if isinstance(term, Mul):
        for arg in term.args:
            if isinstance(arg, Float):
                if arg < atol:
                    return 0
    return term


def _remove_duplicate_freq_tuples(freq_list):
    mod_freq_list = freq_list.copy()
    for tup in freq_list:
        reversed_tup = (tup[1], tup[0])
        if reversed_tup in freq_list:
            mod_freq_list.remove(reversed_tup)
            return _remove_duplicate_freq_tuples(mod_freq_list)
    return mod_freq_list


def _substitute_referenced_freq(tup, freq_dict):
    other_freq = tup[1].free_symbols
    for freq in other_freq:
        if freq in freq_dict.keys():
            new_tup = (tup[0], tup[1].subs(freq, freq_dict[freq]))
            freq_dict[new_tup[0]] = new_tup[1]
            return _substitute_referenced_freq(new_tup, freq_dict)
    return tup


def modify_correlation_btw_freq(correlation_btw_freq):
    mod_correlation_btw_freq = _remove_duplicate_freq_tuples(correlation_btw_freq)
    freq_dict = dict(mod_correlation_btw_freq)
    mod_freq = []
    for freq, term in freq_dict.items():
        tup = _substitute_referenced_freq((freq, term), freq_dict)
        mod_freq.append(tup)
    return mod_freq


def _find_number_of_freq(freq_term):
    if isinstance(freq_term, Add):
        number_of_freq = 0
        for arg in freq_term.args:
            number_of_freq += _find_number_of_freq(arg)
        return number_of_freq

    elif isinstance(freq_term, Mul):
        if len(freq_term.args) == 2:
            if freq_term.args[0] == S.NegativeOne and isinstance(freq_term.args[1], Symbol):
                return 1
            else:
                TypeError("The number of frequencies could not be determined. Make sure that you have not yet inserted the correlations.")
        else:
            TypeError("The number of frequencies could not be determined. Make sure that you have not yet inserted the correlations.")

    elif isinstance(freq_term, Symbol):
        return 1

    else:
        raise TypeError("The number of frequencies could not be determined. Make sure that you have not yet inserted the correlations.")


def find_correlation_btw_freq(sos, incident_freq, initial_state, final_state, number_of_photons):
    if isinstance(sos.expr, Add):
        term_list = [arg for arg in sos.expr.args]
    else:
        term_list = [sos.expr]

    number_of_terms = sos.number_of_terms
    
    subs_dict = {}
    if initial_state == O:
        initial_state_energy = 0
    else:
        initial_state_energy = TransitionFrequency(str(initial_state), real=True)
        subs_dict[initial_state_energy] = 0
    if final_state == O:
        final_state_energy = 0
    else:
        final_state_energy = TransitionFrequency(str(final_state), real=True)
        subs_dict[final_state_energy] = 0
    
    remaining_energy = initial_state_energy - final_state_energy

    denom_list = []
    for term in term_list:
        denom_list += [arg.args[0] for arg in term.args if isinstance(arg, Pow)]

    freq_dict = {}
    for tf in sos.transition_frequencies:
        freq_dict[tf] = {"sum_of_terms": 0, "number_of_freq": None, "included_freq": None}
        
        subs_dict[tf] = 0
        included_freqs = []
        for denom in denom_list:
            if tf in denom.free_symbols:
                freq_term = denom.subs(subs_dict)
                freq_dict[tf]["sum_of_terms"] += freq_term

                freq_term_wo_gamma = freq_term.subs(gamma, 0)
                included_freqs += freq_term_wo_gamma.free_symbols
                if freq_dict[tf]["number_of_freq"] is None:
                    freq_dict[tf]["number_of_freq"] = _find_number_of_freq(freq_term_wo_gamma)
                else:
                    assert freq_dict[tf]["number_of_freq"] == _find_number_of_freq(freq_term_wo_gamma)
        
        freq_dict[tf]["included_freq"] = set(included_freqs)
    correlation_btw_freq = [
            tup for tup in incident_freq if isinstance(tup[1], Symbol) or isinstance(tup[1], Mul) or isinstance(tup[1], Add)
    ]
    specified_freq = list(set(correlation_btw_freq).symmetric_difference(incident_freq))
    mod_correlation_btw_freq = modify_correlation_btw_freq(correlation_btw_freq)
    mod_incident_freq = specified_freq + mod_correlation_btw_freq
    incident_freq_symbols = [tup[0] for tup in mod_incident_freq]
    
    final_correlations = mod_correlation_btw_freq.copy()
    final_dict = {}
    for tf, tf_dict in freq_dict.items():
        lhs = tf_dict["sum_of_terms"]
        rhs = (tf_dict["number_of_freq"] / number_of_photons) * remaining_energy * number_of_terms
        diff = (rhs - lhs).subs(mod_correlation_btw_freq)
        if diff == 0:
            continue
        not_specified_freq = list(tf_dict["included_freq"].difference(incident_freq_symbols))
        if len(not_specified_freq) == 0:
            assert _fix_equal_to_zero(diff.subs(specified_freq)) == 0, (
                    "The law of conversation of energy in photon scattering processes was violated. "
                    "Please check the entered expression and frequencies."
            )                
        elif len(not_specified_freq) == 1:
            corr = list(solveset(diff, not_specified_freq[0], domain=S.Complexes))
            assert len(corr) == 1
            if not_specified_freq[0] in final_dict.keys():
                assert corr[0] == final_dict[not_specified_freq[0]]
            else:
                final_dict[not_specified_freq[0]] = corr[0]
        else:
            raise ValueError("To evaluate the expression, more frequencies must be specified.")
    assert len(final_dict) in [0, 1]
    final_correlations += list(final_dict.items())
    mod_final_correlations = modify_correlation_btw_freq(final_correlations) # in case frequencies are referenced that 
    return mod_final_correlations

if __name__ == "__main__":
    from responsefun.symbols_and_labels import *
    from responsefun.sum_over_states import TransitionMoment, SumOverStates
    from responsefun.test_property import SOS_expressions
    from sympy import UnevaluatedExpr

    alpha_terms = SOS_expressions["alpha"][0]
    alpha_sos = SumOverStates(alpha_terms, [n])
    assert find_correlation_btw_freq(alpha_sos, [(w, 0.5)], O, O, 2) == []

    alpha_sos2 = SumOverStates(
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w_o - 1j*gamma),
            [n],
            perm_pairs=[(op_a, -w_o-1j*gamma), (op_b, w_1+1j*gamma)]
    )
    assert find_correlation_btw_freq(alpha_sos2, [(w_1, 0.5)], O, O, 2) == [(w_o, w_1)]

    beta_term, beta_perm_pairs = SOS_expressions["beta"] 
    beta_sos = SumOverStates(beta_term, [n, k], perm_pairs=beta_perm_pairs)
    assert find_correlation_btw_freq(beta_sos, [(w_1, 0.5), (w_2, 0.5)], O, O, 3) == [(w_o, w_1+w_2)]
    assert find_correlation_btw_freq(beta_sos, [(w_1, 0.5), (w_2, 0.5), (w_o, w_1+w_2)], O, O, 3) == [(w_o, w_1+w_2)]
    assert find_correlation_btw_freq(beta_sos, [(w_1, w_2), (w_o, w_1+w_2)], O, O, 3) == [(w_1, w_2), (w_o, 2*w_2)]

    beta_complex_term, beta_complex_perm_pairs = SOS_expressions["beta_complex"]
    beta_complex_sos = SumOverStates(beta_complex_term, [n, k], perm_pairs=beta_complex_perm_pairs)
    assert find_correlation_btw_freq(beta_complex_sos, [(w_1, 0.5), (w_2, 1.0)], O, O, 3) == [(w_o, w_1+w_2+1j*gamma)]

    gamma_complex_term = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m) * TransitionMoment(m, op_c, k) * TransitionMoment(k, op_d, O)
            / (
                (w_n + UnevaluatedExpr(-w_o - 1j*gamma))
                *(w_m - UnevaluatedExpr(w_2 + 1j*gamma) - UnevaluatedExpr(w_3 + 1j*gamma)) 
                *(w_k - UnevaluatedExpr(w_3 + 1j*gamma))
            )
    )
    gamma_complex_sos = SumOverStates(
            gamma_complex_term, [n, m, k],
            perm_pairs=[(op_a, -w_o-1j*gamma), (op_b, w_1+1j*gamma), (op_c, w_2+1j*gamma), (op_d, w_3+1j*gamma)]
    )
    assert find_correlation_btw_freq(gamma_complex_sos, [(w_1, 0.5), (w_2, 0.5), (w_3, 0.5)], O, O, 4) == [(w_o, w_1+w_2+w_3+2j*gamma)]
    
    w_prime = Symbol("w'")
    rixs_terms = (
            TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
            + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w_prime + 1j*gamma)
    )
    rixs_sos = SumOverStates(rixs_terms, [n])
    assert find_correlation_btw_freq(rixs_sos, [(w, 0.5)], O, f, 2) == [(w_prime, w-1.0*w_f)]

    tpa_terms = (
            (TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - w_1))
            + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - (w_2))
    )
    tpa_sos = SumOverStates(tpa_terms, [n])
    assert find_correlation_btw_freq(tpa_sos, [(w_1, w_2)], O, f, 2) == [(w_1, 0.5*w_f), (w_2, 0.5*w_f)]
    
    threepa_term = (
            TransitionMoment(O, op_a, m) * TransitionMoment(m, op_b, n) * TransitionMoment(n, op_c, f)
            / ((w_n - w_1 - w_2) * (w_m - w_1))
    )
    threepa_perm_pairs = [(op_a, w_1), (op_b, w_2), (op_c, w_3)]
    threepa_sos = SumOverStates(threepa_term, [m, n], perm_pairs=threepa_perm_pairs)
    assert find_correlation_btw_freq(threepa_sos, [(w_1, w_f/3), (w_2, w_1), (w_3, w_1)], O, f, 3) == [(w_1, w_f/3), (w_2, w_f/3), (w_3, w_f/3)]
    assert find_correlation_btw_freq(threepa_sos, [(w_1, w_2), (w_2, w_3), (w_f, w_1+w_2+w_3)], O, f, 3) == [(w_1, w_3), (w_2, w_3), (w_f, 3*w_3)]

    gamma_term = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m) * TransitionMoment(m, op_c, k) * TransitionMoment(k, op_d, O)
            / (
                (w_n + UnevaluatedExpr(-w_o - 1j*gamma))
                *(w_m - UnevaluatedExpr(w_2 + 1j*gamma) - UnevaluatedExpr(w_3 + 1j*gamma)) 
                *(w_k - UnevaluatedExpr(w_3 + 1j*gamma))
            )
    )
    gamma_sos = SumOverStates(
            gamma_term, [n, m, k],
            perm_pairs=[(op_a, -w_o-1j*gamma), (op_b, w_1+1j*gamma), (op_c, w_2+1j*gamma), (op_d, w_3+1j*gamma)]
    )
    assert find_correlation_btw_freq(gamma_sos, [(w_1, 0.5), (w_2, 0.5), (w_3, 0.5)], O, O, 4) == [(w_o, w_1+w_2+w_3+2j*gamma)]

    esp_terms = (
            TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - w_f - w - 1j*gamma)
            + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - w_f + w + 1j*gamma)
    )
    esp_sos = SumOverStates(esp_terms, [n])
    assert find_correlation_btw_freq(esp_sos, [(w, 0.5)], f, f, 2) == []
