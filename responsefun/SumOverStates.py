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

from sympy import Add, Mul, Symbol, latex, solve
from sympy.physics.quantum.state import Bra, Ket
import warnings

from responsefun.ResponseOperator import (
    Moment,
    OneParticleOperator,
    TransitionFrequency,
)
from responsefun.symbols_and_labels import O

ABC = list(string.ascii_uppercase)


def extract_bra_op_ket(expr):
    """Return list of bra*op*ket sequences in a SymPy term."""
    assert isinstance(expr, Mul)
    bok = [Bra, OneParticleOperator, Ket]
    expr_types = [type(term) for term in expr.args]
    ret = [
        list(expr.args[i : i + 3]) for i, k in enumerate(expr_types) if expr_types[i : i + 3] == bok
    ]
    return ret


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


def _validate_expr(expr):
    if isinstance(expr, Add):
        for arg in expr.args:
            return _validate_expr(arg)
    elif isinstance(expr, Mul):
        if not all(not isinstance(arg, Moment) for arg in expr.args):
            raise TypeError(
                "SOS expression must not contain an instance of "
                "<class 'responsefun.ResponseOperator.Moment'>. All transition "
                "moments must be entered as Bra(from_state)*op*Ket(to_state) sequences,"
                "for example by means of 'responsefun.SumOverStates.TransitionMoment'."
            )
    else:
        raise TypeError("SOS expression must be either of type Mul or Add.")


def validate_summation_indices(term, summation_indices):
    if isinstance(term, Add):
        for arg in term.args:
            validate_summation_indices(arg, summation_indices)
    else:
        if len(summation_indices) != len(set(summation_indices)):
            raise ValueError("Same index of summation was specified twice.")

        boks = extract_bra_op_ket(term)
        bras = [bok[0].label[0] for bok in boks]
        kets = [bok[2].label[0] for bok in boks]

        for index in summation_indices:
            if bras.count(index) != 1 or kets.count(index) != 1:
                raise ValueError("Given indices of summation are not correct. "
                                "Each index must be in exactly one bra and one ket.")


def extract_operators_from_sos(term):
    if isinstance(term, Add):
        operators =[]
        operators_unshifted = []
        for arg in term.args:
            ops, ops_unshifted = extract_operators_from_sos(arg)
            operators.append(ops)
            operators_unshifted.append(ops_unshifted)
        are_operators_equal = all(ops == operators_unshifted[0] for ops in operators_unshifted)
        if are_operators_equal:
            all_operators = set()
            for ops in operators:
                all_operators.update(ops)
            return all_operators, operators_unshifted[0]
        else:
            raise ValueError("For the different terms in the SOS expression, "
                             "different operators were found.")
        
    operators = set()
    operators_unshifted = set()
    for arg in term.args:
        if isinstance(arg, OneParticleOperator):
            operators.add(arg)
            operators_unshifted.add(arg.copy_with_new_shifted(False))
    return operators, operators_unshifted
    

def extract_initial_final_from_sos(term, summation_indices):
    if isinstance(term, Add):
        initials_finals = [extract_initial_final_from_sos(arg, summation_indices)
                           for arg in term.args]
        initials_finals = set(initials_finals)
        if len(initials_finals) == 1:
            return initials_finals.pop()
        else:
            raise ValueError("For the different terms in the SOS expression, "
                             "different initial and final states were found.")
    
    boks = extract_bra_op_ket(term)
    bras = []
    kets = []
    for bok in boks:
        bra = bok[0].label[0]
        ket = bok[2].label[0]
        if bra in summation_indices or ket in summation_indices:
            bras.append(bra)
            kets.append(ket)

    indices = set(summation_indices)
    bras = set(bras)
    kets = set(kets)
        
    initial = kets.difference(indices)
    final = bras.difference(indices)
    if len(initial) != 1 or len(final) != 1:
        raise ValueError("Too many undefined excited states were found.")
    return initial.pop(), final.pop()


def _build_sos_via_permutation(term, perm_pairs):
    """Generate a SOS expression via permutation.

    Parameters
    ----------
    term: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
        SymPy expression of the SOS; it can be either the full expression or
        a single term from which the full expression can be generated via permutation.
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
    if isinstance(term, Add):
        sos_expr = 0
        for arg in term.args:
            sos_expr += _build_sos_via_permutation(arg, perm_pairs)
        return sos_expr
    
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
            for j, _ in enumerate(p):
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


class SumOverStates:
    """Class representing a sum-over-states (SOS) expression."""

    def __init__(
        self,
        expr,
        summation_indices,
        *,
        incoming_freqs=None,
        outgoing_freqs=None,
        perm_pairs=None,
        excluded_states=None,
        symmetric=False,
        correlation_btw_freq=None,
    ):
        """
        Parameters
        ----------
        expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
            SymPy expression of the SOS; it can be either the full expression or
            a single term from which the full expression can be generated via permutation.

        summation_indices: list of <class 'sympy.core.symbol.Symbol'>
            List of indices of summation.

        incoming_freqs: list of <class 'sympy.core.symbol.Symbol'>
            List of incoming frequencies.

        outgoing_freqs: list of <class 'sympy.core.symbol.Symbol'>
            List of outgoing frequencies.

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

        symmetric: bool, optional
            Resulting tensor is symmetric;
            by default 'False'.
        """
        _validate_expr(expr)

        if not isinstance(summation_indices, list):
            self._summation_indices = [summation_indices]
        else:
            self._summation_indices = summation_indices.copy()
        assert all(isinstance(index, Symbol) for index in self._summation_indices)
        
        if incoming_freqs is None:
            self._incoming_freqs = []
        elif not isinstance(incoming_freqs, list):
            self._incoming_freqs = [incoming_freqs]
        else:
            self._incoming_freqs = incoming_freqs.copy()
        assert all(isinstance(freq, Symbol) for freq in self._incoming_freqs)

        if outgoing_freqs is None:
            self._outgoing_freqs = []
        elif not isinstance(outgoing_freqs, list):
            self._outgoing_freqs = [outgoing_freqs]
        else:
            self._outgoing_freqs = outgoing_freqs.copy()
        assert all(isinstance(freq, Symbol) for freq in self._outgoing_freqs)

        # no frequency should be specified twice
        assert len(self._incoming_freqs) == len(set(self._incoming_freqs))
        assert len(self._outgoing_freqs) == len(set(self._outgoing_freqs))

        if excluded_states is None:
            self.excluded_states = []
        elif not isinstance(excluded_states, list):
            self.excluded_states = [excluded_states]
        else:
            self.excluded_states = excluded_states.copy()
        assert all(
            isinstance(state, Symbol) or isinstance(state, int) for state in self.excluded_states
        )

        # TODO: automatically check for symmetry
        assert isinstance(symmetric, bool)
        self._symmetric = symmetric

        self._correlation_btw_freq = None
        if correlation_btw_freq is not None:
            warnings.warn(
                "The correlation_btw_freq keyword is deprecated and will be removed.", 
                DeprecationWarning,
            )
            assert isinstance(correlation_btw_freq, list)
            self._correlation_btw_freq = correlation_btw_freq

        if perm_pairs:
            self.expr = _build_sos_via_permutation(expr, perm_pairs)
        else:
            self.expr = expr
        validate_summation_indices(self.expr, self.summation_indices)

        self._operators, self._operators_unshifted = extract_operators_from_sos(self.expr)
        self._components = {op.comp for op in self._operators}
        self._order = len(self._components)
        if self._components.difference(ABC[: self._order]):
            raise ValueError(
                f"It is important that the Cartesian components of an order {self._order} tensor "
                f"be specified as {ABC[:self._order]}."
            )
        self._initial_state, self._final_state = extract_initial_final_from_sos(self.expr, self.summation_indices)
        self._transition_frequencies = [
            TransitionFrequency(index, real=True) for index in self._summation_indices
        ]
        self.expr = self.expr.doit()
        self._is_reversed = False

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
        if self._correlation_btw_freq is not None:
            ret += f"\ncorrelation between frequencies: {self.correlation_btw_freq}"
        return ret

    @property
    def summation_indices(self):
        return self._summation_indices

    @property
    def summation_indices_str(self):
        return [str(si) for si in self._summation_indices]

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def excited_state(self):
        excited_states = set()
        for state in [self.initial_state, self.final_state]:
            if state != O:
                excited_states.add(state)
        if len(excited_states) > 1:
            raise NotImplementedError("Only one excited state that is not a summation index can be contained.")
        return excited_states.pop()

    @property
    def system_energy(self):
        return TransitionFrequency(self.final_state, real=True) - TransitionFrequency(self.initial_state, real=True)

    @property
    def incoming_freqs(self):
        return self._incoming_freqs

    @property
    def outgoing_freqs(self):
        return self._outgoing_freqs

    @property
    def energy_balance(self):
        sum_in = sum(self.incoming_freqs)
        sum_out = sum(self.outgoing_freqs)
        if self._is_reversed:
            balance = sum_in - sum_out + self.system_energy
        else:
            balance = sum_in - sum_out - self.system_energy
        return balance.subs([(TransitionFrequency(O, real=True), 0.0)])

    @property
    def correlation_btw_freq(self):
        if self._correlation_btw_freq is None:
            energy_balance = self.energy_balance
            if isinstance(energy_balance, Add):
                corr = solve(energy_balance, energy_balance.args[0])
                assert len(corr) == 1
                return [(energy_balance.args[0], corr[0])]
            else:
                return []
        else:
            return self._correlation_btw_freq

    @property
    def symmetric(self):
        return self._symmetric
        
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
    def is_hermitian(self):
        return all(op.symmetry == 1 for op in self.operators)

    def check_energy_conservation(self, all_freqs):
        def passed_statement():
            print("Passed energy conservation check.")
            if self.correlation_btw_freq:
                print(f"Found correlation between frequencies: {self.correlation_btw_freq}")
            print()

        print("This SOS expression describes a transition "
              f"from state {self.initial_state} to state {self.final_state}.")
        energy_balance = self.energy_balance
        print(f"The energy balance of the process is {energy_balance}.")
        energy_balance = energy_balance.subs(all_freqs)
        if abs(energy_balance) < 1e-12:
            passed_statement()
            return True
        
        if self.is_hermitian:
            print("Inserting all frequencies does not give zero.\n"
                  "However, since all operators are Hermitian, the process can also be considered "
                  f"in the opposite direction: from state {self.final_state} to state {self.initial_state}.")
            self._is_reversed = True
            energy_balance = self.energy_balance
            print(f"For this process, the energy balance is {energy_balance}.")
            energy_balance = energy_balance.subs(all_freqs)
            if abs(energy_balance) < 1e-12:
                passed_statement()
                return True

        print("Failed energy conservation check. "
              "Please note that transition moments from n to m are defined as <m|op|n>.")
        return False

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
