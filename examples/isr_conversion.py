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

from responsefun.response.response_functions import ResponseFunction
from sympy.physics.quantum.state import Bra, Ket, StateBase
from sympy.physics.quantum.operator import HermitianOperator
import sympy.physics.quantum.operator as qmoperator
from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex, simplify, fraction
from itertools import permutations


# TODO: extract k-mer from multiplication...
def extract_bra_op_ket(expr):
    assert type(expr) == Mul
    bok = [Bra, HermitianOperator, Ket]
    expr_types = [type(term) for term in expr.args]
    ret = [list(expr.args[i:i+3]) for i, k in enumerate(expr_types)
           if expr_types[i:i+3] == bok]
    return ret


class TransitionMoment:
    def __init__(self, from_state, operator, to_state):
        self.expr = Bra(from_state) * operator * Ket(to_state)
    
    def __rmul__(self, other):
        return other * self.expr

    def __mul__(self, other):
        return self.expr * other
    
    def __repr__(self):
        return str(self.expr)


def insert_matrix(expr, M):
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
        for i_d, d in enumerate(denominators):
            if isinstance(d, Add):
                args = d.args
            elif isinstance(d, Symbol):
                args = [d]
            else:
                raise TypeError("denominators")
            for ii, td in enumerate(args):
                if state_label in str(td):
                    if state_label in denominator_match:
                        raise ValueError("")
                    rest = d.subs(td, 0)
                    denominator_match[state_label] = (rest, d)
                    break
        assert len(denominator_match) == 1
        denominator_matches.update(denominator_match)
    assert len(denominator_matches) == len(ketbra_match)
    assert denominator_matches.keys() == ketbra_match.keys()
    
    sub = expr.copy()
    for k in ketbra_match:
        ket, bra = ketbra_match[k]
        freq_argument = denominator_matches[k][0]
        denom_remove = denominator_matches[k][1]
        sub = sub.subs({
            ket: 1,
            bra: (M + freq_argument)**-1,
            denom_remove: 1
        })
    return sub


def insert_isr_transition_moments(expr, operators):
    assert isinstance(operators, list)
    ret = expr.copy()
    for op in operators:
        F = qmoperator.Operator(rf"F({op.label[0]})")
        Fd = adjoint(F)
        ret = ret.subs(Bra(O) * op, Fd)
        ret = ret.subs(op * Ket(O), F)
        # replace the remaining operators with the ISR matrix
        B = qmoperator.Operator(rf"B({op.label[0]})")
        ret = ret.subs(op, B)
    if ret == expr:
        raise ValueError("Could not find any transition moments.")
    return ret


def to_isr_single_term(expr, operators=None):
    if not operators:
        operators = [
            op for op in expr.args if isinstance(op, qmoperator.HermitianOperator)
        ]
    M = qmoperator.Operator("M")
    i1 = insert_isr_transition_moments(expr, operators)
    return insert_matrix(i1, M)


def to_isr(expr, operators=None):
    ret = 0
    if isinstance(expr, Add):
        for s in expr.args:
            ret += to_isr_single_term(s, operators)
    elif isinstance(expr, Mul):
        ret += to_isr_single_term(s, operators)
    return ret


def extra_terms_single_sos(expr, summation_indices, excluded_cases=None):
    """
    :param expr: single SOS term
    :param summation_indices: list of indices of summation
    :param excluded_cases: list of tuples (index, value) with values that are excluded from the summation
    :return: dictionary containing extra terms
    """
    assert type(expr) == Mul        
    bok_list = extract_bra_op_ket(expr)
    special_cases = []
    for index in summation_indices:
        special_cases.append((index, O))
        for bok in bok_list:
            bra, ket = bok[0].label[0], bok[2].label[0]
            if bra == index and (bra, ket) not in special_cases and (ket, bra) not in special_cases:
                special_cases.append((bra, ket))
            elif ket == index and (ket, bra) not in special_cases and (bra, ket) not in special_cases:
                special_cases.append((ket, bra))
    if excluded_cases:
        for case in excluded_cases:
            special_cases.remove(case)
    extra_terms = {}
    for tup in special_cases:
        index, case = tup[0], tup[1]
        if case == O:
            term = expr.subs([tup, (Symbol("w_{}".format(index), real=True), 0)])
            extra_terms[(tup,)] = term
            # find extra terms of extra term
            new_indices = summation_indices.copy()
            new_indices.remove(index)
            if new_indices:
                new_et  = extra_terms_single_sos(term, new_indices, excluded_cases)
                for c, t in new_et.items():
                    if t not in extra_terms.values():
                        extra_terms[(tup,) + c] = t
        else:
            term = expr.subs([tup, (Symbol("w_{}".format(index), real=True), Symbol("w_{}".format(case), real=True))])
            boks = extract_bra_op_ket(term)
            new_term = term
            for bok in boks:
                if bok[0].label[0] == case and bok[2].label[0] == case:
                    new_term = new_term.subs(bok[0]*bok[1]*bok[2], Bra(O)*bok[1]*Ket(O))
            extra_terms[(tup,)] = new_term
            # find extra terms of extra term
            new_indices = summation_indices.copy()
            new_indices.remove(index)
            if new_indices:
                new_et  = extra_terms_single_sos(new_term, new_indices, excluded_cases)
                for c, t in new_et.items():
                    if t not in extra_terms.values():
                        extra_terms[(tup,) + c] = t
    return extra_terms


def compute_extra_terms(expr, summation_indices, excluded_cases=None):
    """
    :param expr: SOS expression
    :param summation_indices: list of indices of summation
    :param excluded_cases: list of tuples (index, value) with values that are excluded from the summation
    :return: list of extra terms
    """
    assert type(summation_indices) == list
    extra_terms_list = []
    if isinstance(expr, Add):
        for single_term in expr.args:
            term_dict = extra_terms_single_sos(single_term, summation_indices, excluded_cases)
            print(term_dict)
            extra_terms_list.append(term_dict)
    elif isinstance(expr, Mul):
        term_dict = extra_terms_single_sos(expr, summation_indices, excluded_cases)
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
                    (Symbol("w_{}".format(ni), real=True), Symbol("w_{}".format(nsi), real=True)) for ni, nsi in subs_list_1
                ]
                subs_list_1 += freq_list
                new_term_1 = term.subs(subs_list_1)
        # convert single (transition) dipole moments into sympy symbols
            boks = extract_bra_op_ket(new_term_1)
            subs_list_2 = []
            for bok in boks:
                bra, ket = bok[0].label[0], bok[2].label[0]
                if bra == O and ket not in summation_indices:
                    mu_symbol = Symbol("{}^{}".format(bok[1].label[0], str(bra)+str(ket)), real=True)
                    subs_list_2.append((bok[0]*bok[1]*bok[2], mu_symbol))
                elif ket == O and bra not in summation_indices:
                    mu_symbol = Symbol("{}^{}".format(bok[1].label[0], str(ket)+str(bra)), real=True)
                    subs_list_2.append((bok[0]*bok[1]*bok[2], mu_symbol))
            new_term_2 = new_term_1.subs(subs_list_2)
            mod_extra_terms.append(new_term_2)
    return mod_extra_terms


def build_sos_via_permutation(term, perm_pairs):
    """
    :param term: single SOS term
    :param perm_pairs: list of tuples (op, freq) to be permuted
    :return: full SOS expression
    """
    assert type(term) == Mul
    assert type(perm_pairs) == list
    operators = [
            op for op in term.args if isinstance(op, qmoperator.HermitianOperator)
    ]
    for op, pair in zip(operators, perm_pairs):
        if op != pair[0]:
            raise ValueError(
                "The pairs (op, freq) must be in the same order as in the entered SOS term."
            )
    perms = list(permutations(perm_pairs))
    sos_expr = term
    for i, p in enumerate(perms):
        if i > 0:
            subs_list = []
            for j, pp in enumerate(p):
                subs_list.append((perms[0][j][0], p[j][0]))
                subs_list.append((perms[0][j][1], p[j][1]))
            new_term = term.subs(subs_list, simultaneous=True)
            sos_expr += new_term
    return sos_expr


def compute_remaining_terms(extra_terms, subs_list=[]):
    num_list = []
    for term in extra_terms:
        num = fraction(term)[0]
        if num not in num_list and -num not in num_list:
            num_list.append(num)
    remaining_terms = 0
    for num in num_list:
        terms_with_num = 0
        for term in extra_terms:
            if fraction(term)[0] == num or fraction(term)[0] == -num:
                terms_with_num += term
        if simplify(terms_with_num.subs(subs_list)) != 0:
            remaining_terms += terms_with_num
    return remaining_terms


# TODO: helper file for general labels and symbols
O, f, n, m, k, gamma = symbols("0, f, n, m, k, \gamma", real=True)
w_f = Symbol("w_{}".format(str(f)), real=True)
w_n = Symbol("w_{}".format(str(n)), real=True)
w_m = Symbol("w_{}".format(str(m)), real=True)
w_k = Symbol("w_{}".format(str(k)), real=True)
w = Symbol("w", real=True)
w_o = Symbol("w_{\sigma}", real=True)
w_1 = Symbol("w_{1}", real=True)
w_2 = Symbol("w_{2}", real=True)
w_3 = Symbol("w_{3}", real=True)

op_a = qmoperator.HermitianOperator(r"\mu_{\alpha}")
op_b = qmoperator.HermitianOperator(r"\mu_{\beta}")
op_c = qmoperator.HermitianOperator(r"\mu_{\gamma}")
op_d = qmoperator.HermitianOperator(r"\mu_{\delta}")

F_alpha = qmoperator.Operator(r"F(\mu_{\alpha})")
F_beta = qmoperator.Operator(r"F(\mu_{\beta})")
F_gamma = qmoperator.Operator(r"F(\mu_{\gamma})")
B = qmoperator.Operator(r"B(\mu_{\beta})")
M = qmoperator.Operator("M")
X = qmoperator.Operator("X")


tm1 = TransitionMoment(O, op_a, f)
tm2 = TransitionMoment(f, op_b, O)

isr_tm = insert_isr_transition_moments(tm1.expr, [op_a])
assert isr_tm == adjoint(F_alpha) * Ket(f)

tm_fn = TransitionMoment(f, op_b, n)
isr_s2s = insert_isr_transition_moments(tm_fn.expr, [op_b])
assert isr_s2s == Bra(f) * qmoperator.Operator(r"B(\mu_{\beta})") * Ket(n)

tm_12 = insert_isr_transition_moments(tm1 * tm2 / (w_f - w), [op_a, op_b])
assert tm_12 == adjoint(F_alpha) * Ket(f) * Bra(f) * F_beta / (w_f - w)

test_cases = {
    "static": {
        "term": adjoint(F_alpha) * Ket(f) * Bra(f) * F_beta / (w_f),
        "ref": adjoint(F_alpha) * (M)**-1 * F_beta
    },
    "freq_neg": {
        "term": adjoint(F_alpha) * Ket(f) * Bra(f) * F_beta / (w_f - w),
        "ref": adjoint(F_alpha) * (M - w)**-1 * F_beta
    },
    "freq_pos": {
        "term": adjoint(F_alpha) * Ket(f) * Bra(f) * F_beta / (w_f + w),
        "ref": adjoint(F_alpha) * (M + w)**-1 * F_beta
    },
    "freq_offset": {
        "term": adjoint(F_alpha) * Ket(f) * Bra(f) * F_beta / (w_f + w - 1),
        "ref": adjoint(F_alpha) * (M + w - 1)**-1 * F_beta
    },
    "tpa_like": {
        "term": adjoint(F_alpha) * Ket(f) * Bra(f) * B * Ket(n) / (w_f - w),
        "ref": adjoint(F_alpha) * (M - w)**-1 * B * Ket(n) 
    },
    "beta_like": {
        "term": adjoint(F_alpha) * Ket(f) * Bra(f) * B * Ket(n) * Bra(n) * F_gamma / ((w_f - w) * (w_n + w)),
        "ref": adjoint(F_alpha) * (M - w)**-1 * B * (M + w)**-1 * F_gamma
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
    # print(latex(ret))


if __name__ == "__main__":
    alpha_sos = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w + 1j*gamma)
    )
    # alpha_extra_terms = compute_extra_terms(alpha_sos, [n])
    # alpha_isr = to_isr(alpha_sos)
    # for term in alpha_extra_terms:
    #     print(term)
    #     alpha_isr += term
    # print(alpha_sos)
    # print(simplify(alpha_isr))

    polarizability = ResponseFunction(r"<<\mu_A;-\mu_B>>", [r"\omega"])
    polarizability.sum_over_states.set_frequencies([0])
    polarizability_isr = to_isr(polarizability.sum_over_states.expression)

    rixs_sos = (
        TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
        + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w - w_f + 1j*gamma)
    )
    # rixs_extra_terms = compute_extra_terms(rixs_sos, [n])
    # rixs_isr = to_isr(rixs_sos)
    # for term in rixs_extra_terms:
    #     print(term)
    #     rixs_isr += term
    # print(rixs_sos)
    # print(simplify(rixs_isr))

    tpa_sos = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - (w_f/2))
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - (w_f/2))
    )
    # tpa_extra_terms = compute_extra_terms(tpa_sos, [n])
    # tpa_isr = to_isr(tpa_sos)
    # for term in tpa_extra_terms:
    #     print(term)
    #     tpa_isr += term
    # print(tpa_sos)
    # print(tpa_isr)

    esp_sos = (
        TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - w_f - w - 1j*gamma)
        + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - w_f + w + 1j*gamma)
    )
    # esp_extra_terms = compute_extra_terms(esp_sos, [n])
    # esp_isr = to_isr(esp_sos)
    # esp_remaining_extra_terms = compute_remaining_terms(esp_extra_terms)
    # print(esp_isr + esp_remaining_extra_terms)

    beta_sos_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o) * (w_k - w_2))
    beta_real_sos = build_sos_via_permutation(
        beta_sos_term, [(op_a, -w_o), (op_b, w_1), (op_c, w_2)]
    )
    # beta_real_extra_terms = compute_extra_terms(beta_real_sos, [n, k])
    # beta_real_isr = to_isr(beta_real_sos)
    # sum_extra_terms_beta = 0
    # for term in beta_real_extra_terms:
    #     print(term)
    #     sum_extra_terms_beta += term
    # print(beta_real_sos)
    # print(beta_real_isr)
    # print(simplify(sum_extra_terms_beta.subs(w_o, w_1+w_2)))
    # print(compute_remaining_terms(beta_real_extra_terms, [(w_o, w_1+w_2)]))

    threepa_sos_term = TransitionMoment(O, op_b, m) * TransitionMoment(m, op_c, n) * TransitionMoment(n, op_d, f)/ ((w_n - w_1 - w_2) * (w_m - w_1))
    threepa_sos = build_sos_via_permutation(
        threepa_sos_term, [(op_b, w_1), (op_c, w_2), (op_d, w_3)]
    )
    # threepa_extra_terms = compute_extra_terms(threepa_sos, [m, n])
    # for term in threepa_extra_terms:
    #     print(term)
    # print(compute_remaining_terms(threepa_extra_terms, [(w_f, w_1+w_2+w_3)]))

    gamma_like_sos = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, f) * TransitionMoment(f, op_c, k) * TransitionMoment(k, op_d, O) / ((w_n - w) * (w_f - w) * (w_k - w))
    # gamma_like_isr = to_isr_single_term(gamma_like_sos)
    # print(gamma_like_isr)
    
    gamma_sos_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m) * TransitionMoment(m, op_c, k) * TransitionMoment(k, op_d, O) / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_k - w_3))
    # gamma_extra_terms = compute_extra_terms(gamma_sos_term, [n, m, k])
    # for term in gamma_extra_terms:
    #     print(term)
    gamma_real_sos = build_sos_via_permutation(
       gamma_sos_term, [(op_a, -w_o), (op_b, w_1), (op_c, w_2), (op_d, w_3)]
    )
    # gamma_real_extra_terms = compute_extra_terms(gamma_real_sos, [n, m, k])
    # for term in gamma_real_extra_terms:
    #     print(term)
    # print(gamma_real_sos)
    # print(compute_remaining_terms(gamma_real_extra_terms, [(w_o, w_1+w_2+w_3)]))


    def accetable_rhs_lhs(term):
        if isinstance(term, adjoint):
            op_string = term.args[0]
        else:
            op_string = term.label
        return "F" in str(op_string)

    # TODO: replace remaining Bra/Ket with ADC Vectors
    # TODO: clever wrapping for terms which can either be Mul or Add
    # TODO: build tree for inversions etc. because one might already need response vectors
    # to form a new LHS/RHS

    def terms_inversion(expr):
        ret = {}
        if isinstance(expr, Add):
            for a in expr.args:
                ret.update(terms_inversion(a))
        elif isinstance(expr, Mul):
            for ii, m in enumerate(expr.args):
                if isinstance(m, Pow):
                    if m.args[1] == -1:
                        tinv = m.args[0]
                        rhs = expr.args[ii + 1]
                        lhs = expr.args[ii - 1]
                        freq = m.args[0].subs(M, 0)
                        ret[m.args[0]] = (lhs, rhs, freq)
                        test = expr.subs(lhs * m, adjoint(X))
                        print(test)
                        print(f"Term {m.args[0]} must be inverted.")
        return ret


    #ret = terms_inversion(polarizability_isr)
    #print(ret)

    #ret = terms_inversion(adjoint(F_alpha) * (M - w)**-1 * B * Ket(n))
    #print(ret)

