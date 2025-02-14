#  Copyright (C) 2023 by the responsefun authors
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

import string
from itertools import permutations

from sympy import Add, Mul, Symbol, latex
from sympy.physics.quantum.state import Bra, Ket

from responsefun.ResponseOperator import (
    Moment,
    OneParticleOperator,
    TransitionFrequency,
)
from responsefun.symbols_and_labels import O

ABC = list(string.ascii_uppercase)


class TransitionMoment:
    """Class representing a transition moment Bra(to_state)*op*Ket(from_state) in a SymPy
    expression."""

    def __init__(self, to_state, operator, from_state, shifted=False):
        if shifted and (from_state == O or to_state == O):
            raise ValueError("Only excited-state-to-excited-state transition moments can be shifted.")
        new_operator = operator.copy_with_new_shifted(shifted)
        self.expr = Bra(to_state) * new_operator * Ket(from_state)

    def __rmul__(self, other):
        return other * self.expr

    def __mul__(self, other):
        return self.expr * other

    def __repr__(self):
        return str(self.expr)


def _build_sos_via_permutation_single(term, perm_pairs):
    """Generate a SOS expression via permutation.

    Parameters
    ----------
    term: <class 'sympy.core.mul.Mul'>
        Single SOS term.
    perm_pairs: list of tuples
        List of (op, freq) pairs whose permutation yields the full SOS expression;
        (op, freq): (<class 'responsefun.ResponseOperator.OneParticleOperator'>,
        <class 'sympy.core.symbol.Symbol'>),
        e.g., [(op_a, -w_o), (op_b, w_1), (op_c, w_2)].

    Returns
    ----------
    <class 'sympy.core.add.Add'>
        Full SOS expression;
        if perm_pairs has only one entry, the returned expression is equal to the entered one,
        and therefore of type <class 'sympy.core.mul.Mul'>.
    """
    assert isinstance(term, Mul)
    assert isinstance(perm_pairs, list)

    # extract operators from the entered SOS term
    operators = tuple([op for op in term.args if isinstance(op, OneParticleOperator)])
    are_operators_shifted = tuple([op.shifted for op in operators])

    # generate SOS term where all operators are unshifted
    unshift_operators = []
    for op in operators:
        if op.shifted:
            op_unshifted = op.copy_with_new_shifted(False)
            unshift_operators.append((op, op_unshifted))
    term_unshifted = term.subs(unshift_operators, simultaneous=True)
    operators_unshifted = tuple([op for op in term_unshifted.args if isinstance(op, OneParticleOperator)])

    # make sure that the (op, freq) pairs are specified in the correct order
    ordered_perm_pairs = []
    for op in operators_unshifted:
        for pair in perm_pairs:
            assert not pair[0].shifted
            if pair[0] == op:
                ordered_perm_pairs.append(pair)
    assert len(ordered_perm_pairs) == len(perm_pairs)

    # generate permutations
    perms = list(permutations(ordered_perm_pairs))
    # successively build up the SOS expression
    sos_expr = term
    for i, p in enumerate(perms):
        if i > 0:
            subs_list = []
            for j, pp in enumerate(p):
                subs_list.append((perms[0][j][0], p[j][0]))
                subs_list.append((perms[0][j][1], p[j][1]))

            new_term_unshifted = term_unshifted.subs(subs_list, simultaneous=True)
            new_operators_unshifted = tuple([op for op in new_term_unshifted.args if isinstance(op, OneParticleOperator)])
            shift_operators = []
            for op, shifted in zip(new_operators_unshifted, are_operators_shifted):
                if shifted:
                    op_shifted = op.copy_with_new_shifted(True)
                    shift_operators.append((op, op_shifted))
            new_term = new_term_unshifted.subs(shift_operators, simultaneous=True)

            sos_expr += new_term
            new_operators = tuple([op for op in new_term.args if isinstance(op, OneParticleOperator)])
            are_new_operators_shifted = tuple([op.shifted for op in new_operators])
            assert are_new_operators_shifted == are_operators_shifted
    return sos_expr


def _build_sos_via_permutation(term, perm_pairs):
    if isinstance(term, Add):
        sos_expr = 0
        for arg in term.args:
            sos_expr += _build_sos_via_permutation_single(arg, perm_pairs)
    else:
        sos_expr = _build_sos_via_permutation_single(term, perm_pairs)
    return sos_expr


class SumOverStates:
    """Class representing a sum-over-states (SOS) expression."""

    def __init__(
        self,
        expr,
        summation_indices,
        *,
        correlation_btw_freq=None,
        perm_pairs=None,
        excluded_states=None,
    ):
        """
        Parameters
        ----------
        expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
            SymPy expression of the SOS; it can be either the full expression or
            a single term from which the full expression can be generated via permutation.

        summation_indices: list of <class 'sympy.core.symbol.Symbol'>
            List of indices of summation.

        correlation_btw_freq: list of tuples, optional
            List that indicates the correlation between the frequencies;
            the tuple entries are either instances of <class 'sympy.core.add.Add'> or
            <class 'sympy.core.symbol.Symbol'>; the first entry is the frequency that can
            be replaced by the second entry, e.g., (w_o, w_1+w_2).

        perm_pairs: list of tuples, optional
            List of (op, freq) pairs whose permutation yields the full SOS expression;
            (op, freq): (<class 'responsefun.ResponseOperator.OneParticleOperator'>,
            <class 'sympy.core.symbol.Symbol'>),
            e.g., [(op_a, -w_o), (op_b, w_1), (op_c, w_2)].

        excluded_states: list of <class 'sympy.core.symbol.Symbol'> or int, optional
            List of states that are excluded from the summation.
            It is important to note that the ground state is represented by the SymPy symbol O,
            while the integer 0
            represents the first excited state.
        """
        if not isinstance(summation_indices, list):
            self._summation_indices = [summation_indices]
        else:
            self._summation_indices = summation_indices.copy()
        assert all(isinstance(index, Symbol) for index in self._summation_indices)

        if correlation_btw_freq:
            assert isinstance(correlation_btw_freq, list)

        if excluded_states is None:
            self.excluded_states = []
        elif not isinstance(excluded_states, list):
            self.excluded_states = [excluded_states]
        else:
            self.excluded_states = excluded_states.copy()
        assert all(
            isinstance(state, Symbol) or isinstance(state, int) for state in self.excluded_states
        )

        if perm_pairs:
            self.expr = _build_sos_via_permutation(expr, perm_pairs)
        else:
            self.expr = expr

        self._operators = set()
        self._components = set()
        if isinstance(self.expr, Add):
            for arg in self.expr.args:
                for a in arg.args:
                    if isinstance(a, OneParticleOperator):
                        self._operators.add(a)
                        for c in a.comp:
                            self._components.add(c)
                    if isinstance(a, Moment):
                        raise TypeError(
                            "SOS expression must not contain an instance of "
                            "<class 'responsefun.ResponseOperator.Moment'>. All transition "
                            "moments must be entered as Bra(from_state)*op*Ket(to_state) sequences,"
                            "for example by means of 'responsefun.SumOverStates.TransitionMoment'>."
                        )
            for index in self._summation_indices:
                for arg in self.expr.args:
                    if Bra(index) not in arg.args or Ket(index) not in arg.args:
                        raise ValueError("Given indices of summation are not correct.")
        elif isinstance(self.expr, Mul):
            for a in self.expr.args:
                if isinstance(a, OneParticleOperator):
                    self._operators.add(a)
                    for c in a.comp:
                        self._components.add(c)
                elif isinstance(a, Moment):
                    raise TypeError(
                        "SOS expression must not contain an instance of "
                        "<class 'responsefun.ResponseOperator.Moment'>. All transition "
                        "moments must be entered as Bra(from_state)*op*Ket(to_state) sequences,"
                        "for example by means of 'responsefun.SumOverStates.TransitionMoment'."
                    )
            for index in self._summation_indices:
                if Bra(index) not in self.expr.args or Ket(index) not in self.expr.args:
                    raise ValueError("Given indices of summation are not correct.")
        else:
            raise TypeError("SOS expression must be either of type Mul or Add.")

        self._order = len(self._components)
        if self._components.difference(ABC[: self._order]):
            raise ValueError(
                f"It is important that the Cartesian components of an order {self._order} tensor "
                f"be specified as {ABC[:self._order]}."
            )

        self._transition_frequencies = [
            TransitionFrequency(index, real=True) for index in self._summation_indices
        ]
        self.correlation_btw_freq = correlation_btw_freq

        self.expr = self.expr.doit()

    def __repr__(self):
        ret = f"Sum over {self._summation_indices}"
        if self.excluded_states:
            ret += f" (excluded: {self.excluded_states}):\n"
        else:
            ret += ":\n"
        if isinstance(self.expr, Add):
            ret += str(self.expr.args[0]) + "\n"
            for term in self.expr.args[1:]:
                ret += "+ " + str(term) + "\n"
            ret = ret[:-1]
        else:
            ret += str(self.expr)
        return ret

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
        if isinstance(self.expr, Add):
            return len(self.expr.args)
        else:
            return 1

    @property
    def latex(self):
        ret = "\\sum_{"
        for index in self._summation_indices:
            ret += f"{index},"
        ret = ret[:-1]
        if self.excluded_states:
            ret += "\\neq"
            for state in self.excluded_states:
                ret += f" {state},"
        ret = ret[:-1]
        ret += "} " + latex(self.expr)
        return ret
