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
from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex
import itertools


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


# TODO: helper file for general labels and symbols
O, f, n, gamma = symbols("0, f, n, \gamma", real=True)
w_f = Symbol("w_{}".format(str(f)), real=True)
w_n = Symbol("w_{}".format(str(n)), real=True)
w = Symbol("w", real=True)

op_a = qmoperator.HermitianOperator(r"\mu_{\alpha}")
op_b = qmoperator.HermitianOperator(r"\mu_{\beta}")

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


alpha_sos_term = TransitionMoment(O, op_a, f) * TransitionMoment(f, op_b, 0) / (w_f - w - 1j*gamma)
alpha_isr_term = to_isr_single_term(alpha_sos_term)
print(latex(alpha_isr_term))


polarizability = ResponseFunction(r"<<\mu_A;-\mu_B>>", [r"\omega"])
polarizability.sum_over_states.set_frequencies([0])
polarizability_isr = to_isr(polarizability.sum_over_states.expression)


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


ret = terms_inversion(polarizability_isr)
print(ret)

ret = terms_inversion(adjoint(F_alpha) * (M - w)**-1 * B * Ket(n))
print(ret)

