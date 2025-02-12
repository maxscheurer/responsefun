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

from sympy import (
    Abs,
    Add,
    Float,
    Integer,
    Mul,
    Pow,
    Symbol,
    adjoint,
    fraction,
    latex,
    simplify,
    zoo,
)
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import Bra, Ket

from responsefun.ResponseOperator import (
    MTM,
    S2S_MTM,
    Moment,
    OneParticleOperator,
    TransitionFrequency,
)
from responsefun.SumOverStates import SumOverStates
from responsefun.symbols_and_labels import M, O


def extract_bra_op_ket(expr):
    """Return list of bra*op*ket sequences in a SymPy term."""
    assert isinstance(expr, Mul)
    bok = [Bra, OneParticleOperator, Ket]
    expr_types = [type(term) for term in expr.args]
    ret = [
        list(expr.args[i : i + 3]) for i, k in enumerate(expr_types) if expr_types[i : i + 3] == bok
    ]
    return ret


def insert_single_moments(expr, summation_indices):
    assert isinstance(expr, Mul)
    boks = extract_bra_op_ket(expr)
    subs_list = []
    for bok in boks:
        # initial state on the ket side and final state on the bra side
        bra, ket = bok[0].label[0], bok[2].label[0]
        op = bok[1]
        if ket == O and bra not in summation_indices:
            from_state = ket
            to_state = bra
            sign = 1.0
        elif bra == O and ket not in summation_indices:
            if op.symmetry == 0:
                from_state = ket
                to_state = bra
                sign = 1.0
            else:
                from_state = bra
                to_state = ket
                if op.symmetry == 1:
                    sign = 1.0
                else:
                    sign = -1.0
        else:
            continue
        mu_symbol = sign * Moment(op.comp, from_state, to_state, op.op_type)
        subs_list.append((bok[0] * bok[1] * bok[2], mu_symbol))
    return expr.subs(subs_list)


def insert_matrix(expr, matrix=Operator("M")):
    """Insert inverse shifted ADC matrix expression."""
    assert isinstance(expr, Mul)
    kb = [Ket, Bra]
    expr_types = [type(term) for term in expr.args]
    ketbra_match = {
        expr.args[i].label[0]: expr.args[i : i + 2]
        for i, k in enumerate(expr_types)
        if expr_types[i : i + 2] == kb  # find Ket-Bra sequence
        and expr.args[i].label[0] == expr.args[i + 1].label[0]  # make sure they have the same state
    }
    denominators = []
    for term in expr.args:
        if isinstance(term, Pow) and term.args[1] < 0:
            assert isinstance(term.args[1], Integer)
            list_to_append = [term.args[0]] * Abs(term.args[1])
            denominators += list_to_append
    denominator_matches = {}
    for state_label in ketbra_match:
        denominator_match = {}
        if state_label == O:
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
                    rest = d.subs(tf, 0)
                    if state_label in denominator_match:
                        denominator_match[state_label].append((rest, d))
                    else:
                        denominator_match[state_label] = [(rest, d)]
                    break
        assert len(denominator_match) == 1
        denominator_matches.update(denominator_match)
    assert len(denominator_matches) == len(ketbra_match)
    assert denominator_matches.keys() == ketbra_match.keys()

    sub = expr.copy()
    for k in ketbra_match:
        ket, bra = ketbra_match[k]
        bra_subs = 1
        denom_dict = {}
        for tup in denominator_matches[k]:
            freq_argument, denom_remove = tup
            bra_subs *= (matrix + freq_argument) ** -1
            denom_dict[denom_remove] = 1
        assert bra_subs != 1
        assert len(denom_dict) > 0
        subs_dict = {}
        subs_dict[ket] = 1
        subs_dict[bra] = bra_subs
        subs_dict.update(denom_dict)
        sub = sub.subs(subs_dict)
    return sub


def insert_isr_transition_moments(expr, operators):
    """Insert vector F of modified transition moments and matrix B of modified excited-states
    transition moments."""
    assert isinstance(expr, Mul)
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
    """Convert a single SOS term to its ADC/ISR formulation by inserting the corresponding ISR
    quantities."""
    assert isinstance(expr, Mul)
    if not operators:
        operators = [op for op in expr.args if isinstance(op, OneParticleOperator)]
    i1 = insert_isr_transition_moments(expr, operators)
    return insert_matrix(i1, M)


def extra_terms_single_sos(expr, summation_indices, excluded_states=None):
    """Determine the additional terms that arise when converting a single SOS term to its ADC/ISR
    formulation.

    Parameters
    ----------
    expr: <class 'sympy.core.mul.Mul'>
        SymPy expression of a single SOS term.

    summation_indices: list of <class 'sympy.core.symbol.Symbol'>
        List of indices of summation.

    excluded_states: list of <class 'sympy.core.symbol.Symbol'> or int, optional
        List of states that are excluded from the summation.
        It is important to note that the ground state is represented by the SymPy symbol O,
        while the integer 0
        represents the first excited state.

    Returns
    ----------
    dict
        Dictionary containing SymPy expressions of computed extra terms
        with the corresponding case as key, e.g., ((n, 0), (m, 0)).
    """
    assert isinstance(expr, Mul)
    if excluded_states is None:
        excluded_states = []
    bok_list = extract_bra_op_ket(expr)
    special_cases = []
    # find special cases
    for index in summation_indices:
        special_cases.append((index, O))
        for bok in bok_list:
            bra, ket = bok[0].label[0], bok[2].label[0]
            if bra == index and (bra, ket) not in special_cases and (ket, bra) not in special_cases:
                special_cases.append((bra, ket))
            elif (
                ket == index and (ket, bra) not in special_cases and (bra, ket) not in special_cases
            ):
                special_cases.append((ket, bra))
    # remove excluded cases
    for state in excluded_states:
        special_cases[:] = [case for case in special_cases if case[1] != state]

    extra_terms = {}
    for tup in special_cases:
        index, case = tup[0], tup[1]
        if case == O:
            term = expr.subs([tup, (TransitionFrequency(index, real=True), 0)])
        else:
            term = expr.subs(
                [tup, (TransitionFrequency(index, real=True), TransitionFrequency(case, real=True))]
            )
            boks = extract_bra_op_ket(term)
            for bok in boks:
                if bok[0].label[0] == case and bok[2].label[0] == case:
                    term = term.subs(bok[0] * bok[1] * bok[2], Bra(O) * bok[1] * Ket(O))
        if term == zoo:
            raise ZeroDivisionError("Extra terms cannot be determined for static SOS expressions.")
        extra_terms[(tup,)] = term
        # find extra terms of extra term
        new_indices = summation_indices.copy()
        new_indices.remove(index)
        if new_indices:
            new_et = extra_terms_single_sos(term, new_indices, excluded_states)
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
        the tuple entries are either instances of <class 'sympy.core.add.Add'> or
        <class 'sympy.core.symbol.Symbol'>;
        the first entry is the frequency that can be replaced by the second entry,
        e.g., (w_o, w_1+w_2).

    Returns
    ----------
    <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'> or 0
        SymPy expression of the extra terms that do not cancel out.
    """
    assert isinstance(extra_terms, list)
    if correlation_btw_freq is None:
        correlation_btw_freq = []
    else:
        assert isinstance(correlation_btw_freq, list)
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


def compute_extra_terms(
    expr,
    summation_indices,
    excluded_states=None,
    correlation_btw_freq=None,
    print_extra_term_dict=False,
):
    """Determine the additional terms that arise when converting an SOS expression to its ADC/ISR
    formulation.

    Parameters
    ----------
    expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
        SymPy expression of the SOS.

    summation_indices: list of <class 'sympy.core.symbol.Symbol'>
            List of indices of summation.

    excluded_states: list of <class 'sympy.core.symbol.Symbol'> or int, optional
        List of states that are excluded from the summation.
        It is important to note that the ground state is represented by the SymPy symbol O,
        while the integer 0
        represents the first excited state.

    correlation_btw_freq: list of tuples, optional
            List that indicates the correlation between the frequencies;
            the tuple entries are either instances of <class 'sympy.core.add.Add'>
            or <class 'sympy.core.symbol.Symbol'>;
            the first entry is the frequency that can be replaced by the second entry, e.g.,
            (w_o, w_1+w_2).

    print_extra_term_dict: bool, optional
        Print dictionary that explains where which additional term comes from,
        by default 'False'.

    Returns
    -----------
    <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'> or 0
        SymPy expression of the extra terms that do not cancel out.
    """
    assert isinstance(summation_indices, list)
    assert isinstance(print_extra_term_dict, bool)

    extra_terms_list = []
    if isinstance(expr, Add):
        terms_list = [arg for arg in expr.args]
    elif isinstance(expr, Mul):
        terms_list = [expr]
    else:
        raise TypeError("SOS expression must be either of type Mul or Add.")
    for it, single_term in enumerate(terms_list):
        term_dict = extra_terms_single_sos(single_term, summation_indices, excluded_states)
        extra_terms_list.append(term_dict)

    mod_extra_terms = []
    for itd, term_dict in enumerate(extra_terms_list):
        if print_extra_term_dict:
            print(f"Additional terms for term {itd+1}:")
        # change remaining indices of summation in extra terms
        for case, term in term_dict.items():
            if print_extra_term_dict:
                print(case, ": ", term)
            new_term_1 = term
            if len(case) != len(summation_indices):
                new_indices = summation_indices.copy()
                for tup in case:
                    new_indices.remove(tup[0])
                subs_list_1 = list(zip(new_indices, summation_indices[: len(new_indices)]))
                freq_list = [
                    (TransitionFrequency(ni, real=True), TransitionFrequency(nsi, real=True))
                    for ni, nsi in subs_list_1
                ]
                subs_list_1 += freq_list
                new_term_1 = term.subs(subs_list_1)
            # convert single (transition) moments to instances of Moment
            new_term_2 = insert_single_moments(new_term_1, summation_indices)
            mod_extra_terms.append(new_term_2)
    return compute_remaining_terms(mod_extra_terms, correlation_btw_freq)


class IsrFormulation:
    """Class representing an ADC/ISR formulation of a response function."""

    def __init__(self, sos, extra_terms=True, print_extra_term_dict=False):
        """
        Parameters
        ----------
        sos: <class 'responsefun.SumOverStates.SumOverStates'>
            SOS expression to be transformed into its ADC/ISR formulation.

        extra_terms: bool, optional
            Compute the additional terms that arise when converting the SOS expression
            to its ADC/ISR formulation; by default 'True'.

        print_extra_term_dict: bool, optional
            Print dictionary explaining the origin of the additional terms, by default 'False'.
        """
        assert isinstance(sos, SumOverStates)
        assert isinstance(extra_terms, bool)
        assert isinstance(print_extra_term_dict, bool)

        if isinstance(sos.expr, Add):
            sos_term_list = [arg for arg in sos.expr.args]
        else:
            sos_term_list = [sos.expr]

        self._main_terms = 0
        for term in sos_term_list:
            mod_term = insert_single_moments(term, sos.summation_indices)
            self._main_terms += to_isr_single_term(mod_term, sos.operators)

        if extra_terms:
            if print_extra_term_dict:
                print("Determining extra terms ...")
            computed_terms = compute_extra_terms(
                sos.expr,
                sos.summation_indices,
                sos.excluded_states,
                sos.correlation_btw_freq,
                print_extra_term_dict,
            )
            self._extra_terms = 0
            if computed_terms == 0:
                pass
            elif isinstance(computed_terms, Add):
                for term in computed_terms.args:
                    self._extra_terms += to_isr_single_term(term, sos.operators)
            else:
                self._extra_terms += to_isr_single_term(computed_terms, sos.operators)
        else:
            self._extra_terms = 0

        self.expr = self._main_terms + self._extra_terms
        self.correlation_btw_freq = sos.correlation_btw_freq

    def __repr__(self):
        ret = ""
        if isinstance(self.expr, Add):
            ret += str(self.expr.args[0]) + "\n"
            for term in self.expr.args[1:]:
                ret += "+ " + str(term) + "\n"
            ret = ret[:-1]
        else:
            ret += str(self.expr)
        return ret

    @property
    def number_of_terms(self):
        if isinstance(self.expr, Add):
            return len(self.expr.args)
        else:
            return 1

    @property
    def number_of_extra_terms(self):
        if self._extra_terms == 0:
            return 0
        elif isinstance(self._extra_terms, Add):
            return len(self._extra_terms.args)
        else:
            return 1

    @property
    def mod_expr(self):
        return self.expr.subs(self.correlation_btw_freq)

    @property
    def latex(self):
        return latex(self.expr)
