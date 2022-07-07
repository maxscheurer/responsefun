#  Copyright (C) 2019 by Maximilian Scheurer
#
#  This file is part of responsefun.
#
#  responsefun is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  responsefun is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with responsefun. If not, see <http:www.gnu.org/licenses/>.
#

from sympy.physics.quantum.state import Bra, Ket, StateBase
from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex, simplify, fraction, zoo, Integer, Float
from sympy.physics.quantum.operator import Operator

from responsefun.symbols_and_labels import *
from responsefun.response_operators import MTM, S2S_MTM, DipoleOperator, DipoleMoment, TransitionFrequency, LeviCivita
from responsefun.sum_over_states import TransitionMoment, SumOverStates
from responsefun.build_tree import build_tree


def extract_bra_op_ket(expr):
    """Return list of bra*op*ket sequences in a SymPy term.
    """
    assert type(expr) == Mul
    bok = [Bra, DipoleOperator, Ket]
    expr_types = [type(term) for term in expr.args]
    ret = [list(expr.args[i:i+3]) for i, k in enumerate(expr_types)
           if expr_types[i:i+3] == bok]
    return ret


def insert_single_dipole_moments(expr, summation_indices):
    assert type(expr) == Mul
    boks = extract_bra_op_ket(expr)
    subs_list = []
    for bok in boks:
        bra, ket = bok[0].label[0], bok[2].label[0]
        if bra == O and ket not in summation_indices:
            mu_symbol = DipoleMoment(bok[1].comp, str(bra), str(ket), bok[1].op_type)
            subs_list.append((bok[0]*bok[1]*bok[2], mu_symbol))
        elif ket == O and bra not in summation_indices:
            mu_symbol = DipoleMoment(bok[1].comp, str(ket), str(bra), bok[1].op_type)
            subs_list.append((bok[0]*bok[1]*bok[2], mu_symbol))
    return expr.subs(subs_list)



def insert_matrix(expr, matrix=Operator("M")):
    """Insert inverse shifted ADC matrix expression.
    """
    assert type(expr) == Mul
    kb = [Ket, Bra]
    expr_types = [type(term) for term in expr.args]
    ketbra_match = {str(expr.args[i].label[0]) : expr.args[i:i+2] for i, k in enumerate(expr_types)
                    if expr_types[i:i+2] == kb  # find Ket-Bra sequence
                    and expr.args[i].label[0] == expr.args[i+1].label[0] # make sure they have the same state
    }
    denominators = [
        x.args[0] for x in expr.args if isinstance(x, Pow) and x.args[1] == -1
    ]
    denominator_matches = {}
    for state_label in ketbra_match:
        denominator_match = {}
        if state_label == "0":
            print("Ground state RI.")
            continue
        for d in denominators:
            if isinstance(d, Add):
                trans_freq = [a for a in d.args if isinstance(a, TransitionFrequency)]
                if len(trans_freq) > 1:
                    raise ValueError("The denominator may contain only one transition frequency.")
            elif isinstance(d, Symbol):
                if isinstance(d, TransitionFrequency):
                    trans_freq = [d]
                else:
                    trans_freq = []
            else:
                raise TypeError("The denominator must be either of type Add or Symbol.")
            for tf in trans_freq:
                if state_label == tf.state:
                    if state_label in denominator_match:
                        raise ValueError("{} was found twice in the SOS expression.".format(tf))
                    rest = d.subs(tf, 0)
                    denominator_match[state_label] = (rest, d)
                    break
        assert len(denominator_match) == 1
        denominator_matches.update(denominator_match)
    assert len(denominator_matches) == len(ketbra_match)
    assert denominator_matches.keys() == ketbra_match.keys()
    
    sub = expr.copy()
    for k in ketbra_match:
        ket, bra = ketbra_match[k]
        freq_argument, denom_remove = denominator_matches[k]
        sub = sub.subs({
            ket: 1,
            bra: (matrix + freq_argument)**-1,
            denom_remove: 1
        })
    return sub


def insert_isr_transition_moments(expr, operators):
    """Insert vector F of modified transition moments and matrix B of modified excited-states transition moments.
    """
    assert type(expr) == Mul
    assert isinstance(operators, list)
    ret = expr.copy()
    for op in operators:
        F = MTM(op.comp, op.op_type)
        Fd = adjoint(F)
        ret = ret.subs(Bra(O) * op, Fd)
        ret = ret.subs(op * Ket(O), F)
        # replace the remaining operators with the ISR matrix
        B = S2S_MTM(op.comp, op.op_type)
        ret = ret.subs(op, B)
    if ret == expr:
        print("Term contains no transition moment.")
    return ret


def to_isr_single_term(expr, operators=None):
    """Convert a single SOS term to its ADC/ISR formulation by inserting the corresponding ISR quantities.
    """
    assert type(expr) == Mul
    if not operators:
        operators = [
            op for op in expr.args if isinstance(op, DipoleOperator)
        ]
    i1 = insert_isr_transition_moments(expr, operators)
    M = Operator("M")
    return insert_matrix(i1, M)


def extra_terms_single_sos(expr, summation_indices, excluded_cases=None):
    """Determine the additional terms that arise when converting a single SOS term to its ADC/ISR formulation.
    
    Parameters
    ----------
    expr: <class 'sympy.core.mul.Mul'>
        SymPy expression of a single SOS term.
    
    summation_indices: list of <class 'sympy.core.symbol.Symbol'>
        List of indices of summation.
    
    excluded_cases: list of tuples, optional
        List of (summation_index, value) pairs with values that are excluded from the summation
        (summation_index, value): (<class 'sympy.core.symbol.Symbol'>, int).

    Returns
    ----------
    dict
        Dictionary containing SymPy expressions of computed extra terms with the corresponding case as key, e.g., ((n, 0), (m, 0)).
    """
    assert type(expr) == Mul        
    bok_list = extract_bra_op_ket(expr)
    special_cases = []
    # find special cases
    for index in summation_indices:
        special_cases.append((index, O))
        for bok in bok_list:
            bra, ket = bok[0].label[0], bok[2].label[0]
            if bra == index and (bra, ket) not in special_cases and (ket, bra) not in special_cases:
                special_cases.append((bra, ket))
            elif ket == index and (ket, bra) not in special_cases and (bra, ket) not in special_cases:
                special_cases.append((ket, bra))
    #remove excluded cases
    if excluded_cases:
        for case in excluded_cases:
            special_cases.remove(case)
    extra_terms = {}
    for tup in special_cases:
        index, case = tup[0], tup[1]
        if case == O:
            term = expr.subs([tup, (TransitionFrequency(str(index), real=True), 0)])
        else:
            term = expr.subs([tup, (TransitionFrequency(str(index), real=True), TransitionFrequency(str(case), real=True))])
            boks = extract_bra_op_ket(term)
            for bok in boks:
                if bok[0].label[0] == case and bok[2].label[0] == case:
                    term = term.subs(bok[0]*bok[1]*bok[2], Bra(O)*bok[1]*Ket(O))
        if term == zoo:
            raise ZeroDivisionError("Extra terms cannot be determined for static SOS expressions.")
        extra_terms[(tup,)] = term
        # find extra terms of extra term
        new_indices = summation_indices.copy()
        new_indices.remove(index)
        if new_indices:
            new_et  = extra_terms_single_sos(term, new_indices, excluded_cases)
            for c, t in new_et.items():
                if t not in extra_terms.values():
                    extra_terms[(tup,) + c] = t
    return extra_terms


def compute_remaining_terms(extra_terms, correlation_btw_freq=None):
    """Sort the extra terms by numerators before simplifying them.
    
    Parameters
    ----------
    extra_terms: list
        List containing extra terms.

    correlation_btw_freq: list of tuples, optional
        List that indicates the correlation between the frequencies;
        the tuple entries are either instances of <class 'sympy.core.add.Add'> or <class 'sympy.core.symbol.Symbol'>;
        the first entry is the frequency that can be replaced by the second entry, e.g., (w_o, w_1+w_2).

    Returns
    ----------
    <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'> or 0
        SymPy expression of the extra terms that do not cancel out.
    """
    assert type(extra_terms) == list
    if correlation_btw_freq is None:
        correlation_btw_freq = []
    else:
        assert type(correlation_btw_freq) == list
    num_dict = {}
    for term in extra_terms:
        num = fraction(term)[0]
        mod_num = num
        for arg in num.args:
            if isinstance(arg, Integer) or isinstance(arg, Float):
                mod_num = mod_num.subs(arg, 1)
        if mod_num not in num_dict:
            num_dict[mod_num] = term
        else:
            num_dict[mod_num] += term
    remaining_terms = 0
    for term in num_dict.values():
        if simplify(term.subs(correlation_btw_freq)) != 0:
            remaining_terms += term
    return remaining_terms


def compute_extra_terms(expr, summation_indices, excluded_cases=None, correlation_btw_freq=None, print_extra_term_dict=False):
    """Determine the additional terms that arise when converting an SOS expression to its ADC/ISR formulation.

    Parameters
    ----------
    expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
        SymPy expression of the SOS.

    summation_indices: list of <class 'sympy.core.symbol.Symbol'>
            List of indices of summation.

    excluded_cases: list of tuples, optional
            List of (summation_index, value) pairs with values that are excluded from the summation
            (summation_index, value): (<class 'sympy.core.symbol.Symbol'>, int).
            
    correlation_btw_freq: list of tuples, optional
            List that indicates the correlation between the frequencies;
            the tuple entries are either instances of <class 'sympy.core.add.Add'> or <class 'sympy.core.symbol.Symbol'>;
            the first entry is the frequency that can be replaced by the second entry, e.g., (w_o, w_1+w_2).

    print_extra_term_dict: bool, optional
        Print dictionary that explains where which additional term comes from,
        by default 'False'.

    Returns
    -----------
    <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'> or 0
        SymPy expression of the extra terms that do not cancel out.
    """
    assert type(summation_indices) == list
    assert type(print_extra_term_dict) == bool

    extra_terms_list = []
    if isinstance(expr, Add):
        terms_list = [arg for arg in expr.args]
    elif isinstance(expr, Mul):
        terms_list = [expr]
    else:
        raise TypeError("SOS expression must be either of type Mul or Add.")
    for single_term in terms_list:
        term_dict = extra_terms_single_sos(single_term, summation_indices, excluded_cases)
        if print_extra_term_dict:
            print(term_dict)
        extra_terms_list.append(term_dict)
        
    mod_extra_terms = []
    for term_dict in extra_terms_list:
        # change remaining indices of summation in extra terms
        for case, term in term_dict.items():
            new_term_1 = term
            if len(case) != len(summation_indices):
                new_indices = summation_indices.copy()
                for tup in case:
                    new_indices.remove(tup[0])
                subs_list_1 = list(zip(new_indices, summation_indices[:len(new_indices)]))
                freq_list = [
                    (TransitionFrequency(str(ni), real=True), TransitionFrequency(str(nsi), real=True)) for ni, nsi in subs_list_1
                ]
                subs_list_1 += freq_list
                new_term_1 = term.subs(subs_list_1)
        # convert single (transition) dipole moments to instances of DipoleMoment
            new_term_2 = insert_single_dipole_moments(new_term_1, summation_indices)
            mod_extra_terms.append(new_term_2)
    return compute_remaining_terms(mod_extra_terms, correlation_btw_freq)


def to_isr(sos, extra_terms=True, print_extra_term_dict=False):
    """Convert an SOS expression to its ADC/ISR formulation.
    
    Parameters
    ----------
    sos: <class 'responsetree.sum_over_states.SumOverStates'>
    
    extra_terms: bool, optional
        Compute the additional terms that arise when converting the SOS expression to its ADC/ISR formulation;
        by default 'True'.

    print_extra_term_dict: bool, optional
        Print dictionary explaining the origin of the additional terms,
        by default 'False'.

    Returns
    ----------
    <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
    """
    assert isinstance(sos, SumOverStates)
    assert type(extra_terms) == bool

    if isinstance(sos.expr, Add):
        terms_list = [arg for arg in sos.expr.args]
    else:
        terms_list = [sos.expr]

    mod_expr = 0
    for term in terms_list:
        mod_expr += insert_single_dipole_moments(term, sos.summation_indices)

    if extra_terms:
        mod_expr += compute_extra_terms(sos.expr, sos.summation_indices, sos.excluded_cases, sos.correlation_btw_freq, print_extra_term_dict)

    ret = 0
    if isinstance(mod_expr, Add):
        for s in mod_expr.args:
            ret += to_isr_single_term(s, sos.operators)
    else:
        ret += to_isr_single_term(mod_expr, sos.operators)
    return ret


tm1 = TransitionMoment(O, op_a, f)
tm2 = TransitionMoment(f, op_b, O)

isr_tm = insert_isr_transition_moments(tm1.expr, [op_a])
assert isr_tm == adjoint(F_A) * Ket(f)

tm_fn = TransitionMoment(f, op_b, n)
isr_s2s = insert_isr_transition_moments(tm_fn.expr, [op_b])
assert isr_s2s == Bra(f) * B_B * Ket(n)

tm_12 = insert_isr_transition_moments(tm1 * tm2 / (w_f - w), [op_a, op_b])
assert tm_12 == adjoint(F_A) * Ket(f) * Bra(f) * F_B / (w_f - w)

test_cases = {
    "static": {
        "term": adjoint(F_A) * Ket(f) * Bra(f) * F_B / (w_f),
        "ref": adjoint(F_A) * (M)**-1 * F_B
    },
    "freq_neg": {
        "term": adjoint(F_A) * Ket(f) * Bra(f) * F_B / (w_f - w),
        "ref": adjoint(F_A) * (M - w)**-1 * F_B
    },
    "freq_pos": {
        "term": adjoint(F_A) * Ket(f) * Bra(f) * F_B / (w_f + w),
        "ref": adjoint(F_A) * (M + w)**-1 * F_B
    },
    "freq_offset": {
        "term": adjoint(F_A) * Ket(f) * Bra(f) * F_B / (w_f + w - 1),
        "ref": adjoint(F_A) * (M + w - 1)**-1 * F_B
    },
    "tpa_like": {
        "term": adjoint(F_A) * Ket(f) * Bra(f) * B_B * Ket(n) / (w_f - w),
        "ref": adjoint(F_A) * (M - w)**-1 * B_B * Ket(n) 
    },
    "beta_like": {
        "term": adjoint(F_A) * Ket(f) * Bra(f) * B_B * Ket(n) * Bra(n) * F_C / ((w_f - w) * (w_n + w)),
        "ref": adjoint(F_A) * (M - w)**-1 * B_B * (M + w)**-1 * F_C
    }
}

for case in test_cases:
    tc = test_cases[case]
    term = tc["term"]
    ref = tc["ref"]
    ret = insert_matrix(term, M)
    if ret != ref:
        raise AssertionError(f"Test {case} failed:"
                            " ref = {ref}, ret = {ret}")
    #print(latex(ret))


if __name__ == "__main__":
    alpha_terms = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w + 1j*gamma)
    )
    alpha_sos = SumOverStates(alpha_terms, [n])
    #alpha_isr = to_isr(alpha_sos)
    #print(alpha_sos.expr)
    #print(alpha_isr)
    #build_tree(alpha_isr)
    
    rixs_terms = (
        TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
        + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w - w_f + 1j*gamma)
    )
    rixs_sos = SumOverStates(rixs_terms, [n])
    #rixs_isr = to_isr(rixs_sos)
    #print(rixs_sos.expr)
    #print(rixs_isr)
    #build_tree(rixs_isr)

    rixs_term_short = rixs_terms.args[0]
    rixs_sos_short = SumOverStates(rixs_term_short, [n])
    #rixs_isr_short = to_isr(rixs_sos_short)
    #print(rixs_sos_short.expr)
    #print(rixs_isr_short)
    #build_tree(rixs_isr_short)
    #print(compute_extra_terms(rixs_term_short, [n], print_extra_term_dict=True))

    tpa_terms = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - (w_f/2))
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - (w_f/2))
    )
    tpa_sos = SumOverStates(tpa_terms, [n])
    #tpa_isr = to_isr(tpa_sos)
    #print(tpa_sos.expr)
    #print(tpa_isr)
    #build_tree(tpa_isr)

    esp_terms = (
        TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - w_f - w - 1j*gamma)
        + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - w_f + w + 1j*gamma)
    )
    esp_sos = SumOverStates(esp_terms, [n])
    #esp_isr = to_isr(esp_sos)
    #print(esp_sos.expr)
    #print(esp_isr)
    #build_tree(esp_isr)

    beta_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o) * (w_k - w_2))
    #beta_sos = SumOverStates(beta_term, [n, k], [(w_o, w_1+w_2)], [(op_a, -w_o), (op_b, w_1), (op_c, w_2)])
    beta_sos_test = SumOverStates(beta_term, [n, k])
    #beta_isr = to_isr(beta_sos_test, extra_terms=False)
    #print(beta_sos.expr)
    #print(beta_isr)
    #build_tree(beta_isr)

    beta_complex_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o - 1j*gamma) * (w_k - w_2 - 1j*gamma))
    beta_complex_sos = SumOverStates(beta_complex_term, [n, k], correlation_btw_freq=[(w_o, w_1+w_2+1j*gamma)], perm_pairs=[(op_a, -w_o-1j*gamma), (op_b, w_1+1j*gamma), (op_c, w_2+1j*gamma)])
    #print(beta_complex_sos.expr)
    #extra_terms_beta = compute_extra_terms(beta_complex_sos.expr, beta_complex_sos.summation_indices, correlation_btw_freq=beta_complex_sos.correlation_btw_freq)
    #print(extra_terms_beta)
    #beta_complex_isr = to_isr(beta_complex_sos)

    threepa_term = TransitionMoment(O, op_a, m) * TransitionMoment(m, op_b, n) * TransitionMoment(n, op_c, f) / ((w_n - w_1 - w_2) * (w_m - w_1))
    #threepa_term_test = TransitionMoment(O, op_a, m) * TransitionMoment(m, op_b, n) * TransitionMoment(n, op_c, f) / ((w_n - w_f/3 - w_f/3) * (w_m - w_f/3))
    #threepa_term_test = TransitionMoment(O, op_a, m) * TransitionMoment(m, op_b, n) * TransitionMoment(n, op_c, f) / ((w_n - w - w) * (w_m - w))
    threepa_sos = SumOverStates(threepa_term, [m, n], correlation_btw_freq=[(w_1, (w_f/3)), (w_2, w_1), (w_3, w_1)], perm_pairs=[(op_a, w_1), (op_b, w_2), (op_c, w_3)])
    #threepa_sos = SumOverStates(threepa_term_test, [m, n], perm_pairs=[(op_a, w), (op_b, w), (op_c, w)])
    #threepa_terms_mod = threepa_sos.expr
    #threepa_terms_mod = threepa_sos.expr.subs([(w_1, w), (w_2, w), (w_3, w)])
    #for arg in threepa_sos.expr.args:
    #    print(arg)
    threepa_extra_terms = compute_extra_terms(threepa_sos.expr, [m, n], correlation_btw_freq=threepa_sos.correlation_btw_freq, print_extra_term_dict=True)
    for arg in threepa_extra_terms.args:
        print("here: ", arg)
    #threepa_isr = to_isr(threepa_sos)
    #print(threepa_sos.expr)
    #print(len(threepa_isr.args))
    #for arg in threepa_isr.args:
    #    print(arg)

    gamma_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m) * TransitionMoment(m, op_c, k) * TransitionMoment(k, op_d, O) / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_k - w_3))
    gamma_sos = SumOverStates(gamma_term, [n, m, k], correlation_btw_freq=[(w_o, w_1+w_2+w_3)], perm_pairs=[(op_a, -w_o), (op_b, w_1), (op_c, w_2), (op_d, w_3)])
    #gamma_isr = to_isr(gamma_sos)
    #print(gamma_sos.expr)
    #print(len(gamma_isr.args))
    #for arg in gamma_isr.args:
    #    print(arg)
    #extra_terms_gamma = compute_extra_terms(gamma_sos.expr, gamma_sos.summation_indices, correlation_btw_freq=gamma_sos.correlation_btw_freq)
    #print(len(extra_terms_gamma.args))
    #for arg in extra_terms_gamma.args:
    #    print(arg)

    gamma_test_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) * TransitionMoment(O, op_c, m) * TransitionMoment(m, op_d, O) / ((w_n - w_o) * (-w_2 - w_3) * (w_m - w_3))
    #print(gamma_test_term)
    #gamma_test_sos = SumOverStates(gamma_test_term, [n, m], [(w_o, w_1+w_2+w_3)])
    #gamma_test_isr = to_isr(gamma_test_sos, False)
    #print(gamma_test_isr)

    epsilon = LeviCivita()
    mcd_term1 = (
            -1.0 * epsilon
            * TransitionMoment(O, opm_b, k) * TransitionMoment(k, op_c, f) * TransitionMoment(f, op_a, O)
            / w_k
    )
    mcd_term1_sos = SumOverStates(mcd_term1, [k], excluded_cases=[(k, O)])
    #mcd_term1_isr = to_isr(mcd_term1_sos)
    #print(mcd_term1_sos.expr)
    #print(mcd_term1_isr)
