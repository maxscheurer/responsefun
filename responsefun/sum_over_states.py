from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex, simplify, fraction
from sympy.physics.quantum.state import Bra, Ket, StateBase
from responsefun.response_operators import DipoleOperator, DipoleMoment, LeviCivita
from itertools import permutations
import string

from responsefun.symbols_and_labels import *

ABC = list(string.ascii_uppercase)

class TransitionMoment:
    """
    Class representing a transition moment Bra(from_state)*op*Ket(to_state) in a SymPy expression.
    """
    def __init__(self, from_state, operator, to_state):
        self.expr = Bra(from_state) * operator * Ket(to_state)

    def __rmul__(self, other):
        return other * self.expr

    def __mul__(self, other):
        return self.expr * other

    def __repr__(self):
        return str(self.expr)


def _build_sos_via_permutation(term, perm_pairs):
    """Generate a SOS expression via permutation.
    Parameters
    ----------
    term: <class 'sympy.core.mul.Mul'>
        Single SOS term.
    perm_pairs: list of tuples
        List of (op, freq) pairs whose permutation yields the full SOS expression;
        (op, freq): (<class 'responsetree.response_operators.DipoleOperator'>, <class 'sympy.core.symbol.Symbol'>),
        e.g., [(op_a, -w_o), (op_b, w_1), (op_c, w_2)].

    Returns
    ----------
    <class 'sympy.core.add.Add'>
        Full SOS expression;
        if perm_pairs has only one entry, the returned expression is equal to the entered one,
        and therefore of type <class 'sympy.core.mul.Mul'>.
    """
    assert type(term) == Mul
    assert type(perm_pairs) == list

    # extract operators from the entered SOS term
    operators = [
        op for op in term.args if isinstance(op, DipoleOperator)
    ]
    # check that the (op, freq) pairs are specified in the correct order
    for op, pair in zip(operators, perm_pairs):
        if op != pair[0]:
            raise ValueError(
                "The pairs (op, freq) must be in the same order as in the entered SOS term."
            )
    # generate permutations
    perms = list(permutations(perm_pairs))
    # successively build up the SOS expression
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
    """
    Class representing a sum-over-states (SOS) expression. 
    """
    
    def __init__(self, expr, summation_indices, *, correlation_btw_freq=None, perm_pairs=None, excluded_cases=None):
        """
        Parameters
        ----------
        expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
            SymPy expression of the SOS;
            it can be either the full expression or a single term from which the full expression can be generated via permutation.

        summation_indices: list of <class 'sympy.core.symbol.Symbol'>
            List of indices of summation.

        correlation_btw_freq: list of tuples, optional
            List that indicates the correlation between the frequencies;
            the tuple entries are either instances of <class 'sympy.core.add.Add'> or <class 'sympy.core.symbol.Symbol'>;
            the first entry is the frequency that can be replaced by the second entry, e.g., (w_o, w_1+w_2).

        perm_pairs: list of tuples, optional
            List of (op, freq) pairs whose permutation yields the full SOS expression;
            (op, freq): (<class 'responsetree.response_operators.DipoleOperator'>, <class 'sympy.core.symbol.Symbol'>),
            e.g., [(op_a, -w_o), (op_b, w_1), (op_c, w_2)].

        excluded_cases: list of tuples, optional
            List of (summation_index, value) pairs with values that are excluded from the summation
            (summation_index, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.symbol.Symbol'>).
        """
        assert isinstance(summation_indices, list)
        if correlation_btw_freq:
            assert isinstance(correlation_btw_freq, list)

        if excluded_cases is None:
            sos.excluded_cases = []
        else:
            sos.excluded_cases = excluded_cases
        assert isinstance(sos.excluded_cases, list)
        for case in sos.excluded_cases:
            assert isinstance(case[0], Symbol) and isinstance(case[1], Symbol)

        if isinstance(expr, Add):
            self._operators = []
            self._components = []
            for arg in expr.args:
                for a in arg.args:
                    if isinstance(a, DipoleOperator) and a not in self._operators:
                        self._operators.append(a)
                        for c in a.comp:
                            self._components.append(c)
                    if isinstance(a, DipoleMoment):
                        raise TypeError(
                                "SOS expression must not contain an instance of <class 'responsetree.response_operators.DipoleMoment'>. "
                                "All transition moments must be entered as Bra(from_state)*op*Ket(to_state) sequences, for example by "
                                "means of <class 'responsetree.sum_over_states.TransitionMoment'>."
                        )
            self._components.sort()
            for index in summation_indices:
                for arg in expr.args:
                    if Bra(index) not in arg.args or Ket(index) not in arg.args:
                        raise ValueError("Given indices of summation are not correct.")
        elif isinstance(expr, Mul):
            self._operators = [a for a in expr.args if isinstance(a, DipoleOperator)]
            self._components = []
            for a in expr.args:
                if isinstance(a, DipoleOperator):
                    for c in a.comp:
                        self._components.append(c)
                elif isinstance(a, DipoleMoment):
                    raise TypeError(
                            "SOS expression must not contain an instance of <class 'responsetree.response_operators.DipoleMoment'>. "
                            "All transition moments must be entered as Bra(from_state)*op*Ket(to_state) sequences, for example by "
                            "means of <class 'responsetree.sum_over_states.TransitionMoment'>."
                    )
            self._components.sort()
            for index in summation_indices:
                if Bra(index) not in expr.args or Ket(index) not in expr.args:
                    raise ValueError("Given indices of summation are not correct.")
        else:
            raise TypeError("SOS expression must be either of type Mul or Add.")

        self._order = len(self._components)
        if self._components != ABC[:self._order]:
            raise ValueError(
                    f"It is important that the Cartesian components of an order {self._order} tensor be specified as {ABC[:self._order]}."
            )

        self._summation_indices = summation_indices
        self._transition_frequencies = [TransitionFrequency(str(index), real=True) for index in self._summation_indices]
        self.correlation_btw_freq = correlation_btw_freq
        
        if perm_pairs:
            self.expr = _build_sos_via_permutation(expr, perm_pairs)
        else:
            self.expr = expr
        self.expr = self.expr.doit()

    @property
    def summation_indices(self):
        return self._summation_indices

    @property
    def summation_indices_str(self):
        return [str(si) for si in self._summation_indices]

    @property
    def operators(self):
        return self._operators

    @property
    def operator_types(self):
        return set([op.op_type for op in self._operators])

    @property
    def components(self):
        return self._components
    
    @property
    def transition_frequencies(self):
        return self._transition_frequencies

    @property
    def order(self):
        return self._order

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
    alpha_sos = SumOverStates(alpha_sos_expr, [n])
    #print(alpha_sos.expr, alpha_sos.summation_indices, alpha_sos.transition_frequencies, alpha_sos.order, alpha_sos.operators, alpha_sos.correlation_btw_freq)
    
    beta_sos_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o) * (w_k - w_2))
    beta_sos = SumOverStates(beta_sos_term, [n, k], perm_pairs=[(op_a, -w_o), (op_b, w_1), (op_c, w_2)])
    #print(beta_sos.expr.args)
    #print(beta_sos.summation_indices)
    #print(beta_sos.transition_frequencies, beta_sos.order, beta_sos.operators, beta_sos.number_of_terms)
    #print(beta_sos.latex)

    gamma_extra_terms = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) * TransitionMoment(O, op_c, m) * TransitionMoment(m, op_d, O)
        / ((w_n - w_o) * (w_m - w_3) * (w_m + w_2))
    )
