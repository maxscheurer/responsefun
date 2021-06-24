from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex, simplify, fraction
from sympy.physics.quantum.state import Bra, Ket, StateBase
from itertools import permutations

from responsetree.symbols_and_labels import *
from responsetree.transition_moments import TransitionMoment


def _build_sos_via_permutation(term, perm_pairs):
    """
    :param term: single SOS term
    :param perm_pairs: list of tuples (op, freq) to be permuted
    :return: full SOS expression
    """
    assert type(term) == Mul
    assert type(perm_pairs) == list
    operators = [
            op for op in term.args if isinstance(op, DipoleOperator)
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


class SumOverStates:
    
    def __init__(self, expr, summation_indices, correlation_btw_freq=None, perm_pairs=None):
        """
        Class representing sum over states (SOS)
        
        :param expr: sympy expression of the SOS;
            it can be either the full expression or a single term from which the full expression can be generated via permutation
        :param summation_indices: list of indices of summation (sympy symbols)
        :param correlation_btw_freq: list of tuples that indicates the correlation between the frequencies (sympy symbols);
            the first entry is the frequency that can be replaced by the second entry e.g. (w_o, w_1+w_2)
        :param perm_pairs: list of tuples (op, freq) to be permuted
        """
        assert type(summation_indices) == list
        if correlation_btw_freq:
            assert type(correlation_btw_freq) == list

        if isinstance(expr, Add):
            self.operators = [op for op in expr.args[0].args if isinstance(op, DipoleOperator)]
            for index in summation_indices:
                if Bra(index) not in expr.args[0].args or Ket(index) not in expr.args[0].args:
                    raise ValueError("Given indices of summation are not correct.")
        elif isinstance(expr, Mul):
            self.operators = [op for op in expr.args if isinstance(op, DipoleOperator)]
            for index in summation_indices:
                if Bra(index) not in expr.args or Ket(index) not in expr.args:
                    raise ValueError("Given indices of summation are not correct.")
        else:
            raise TypeError("SOS expression must be either of type Mul or Add.")

        self.summation_indices = summation_indices
        self.transition_frequencies = [Symbol("w_{{{}}}".format(index)) for index in self.summation_indices]
        self.correlation_btw_freq = correlation_btw_freq
        
        if perm_pairs:
            self.expr = _build_sos_via_permutation(expr, perm_pairs)
        else:
            self.expr = expr

    @property
    def order(self):
        return len(self.summation_indices) + 1

    @property
    def number_of_terms(self):
        return len(self.expr.args)

    @property
    def latex(self):
        return latex(self.expr)


if __name__ == "__main__":
    alpha_sos_expr = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
            + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w + 1j*gamma)
        )

    alpha_sos = SumOverStates(alpha_sos_expr, [n], [w])
    #print(alpha_sos.expr, alpha_sos.summation_indices, alpha_sos.transition_frequencies, alpha_sos.order, alpha_sos.operators)


    beta_sos_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o) * (w_k - w_2))
    beta_sos = SumOverStates(beta_sos_term, [n, k], [w_1, w_2], [(op_a, -w_o), (op_b, w_1), (op_c, w_2)])
    #print(beta_sos.expr)
    #print(beta_sos.summation_indices)
    #print(beta_sos.transition_frequencies, beta_sos.order, beta_sos.operators, beta_sos.number_of_terms)
    #print(beta_sos.latex)
