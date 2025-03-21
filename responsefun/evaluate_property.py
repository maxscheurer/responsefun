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
import warnings
from collections import namedtuple
from itertools import combinations_with_replacement, permutations, product

import numpy as np
from adcc import AmplitudeVector
from adcc.workflow import construct_adcmatrix
from respondo.cpp_algebra import ResponseVector as RV
from respondo.solve_response import solve_response
from sympy import (
    Add,
    Float,
    I,
    Integer,
    Mul,
    Number,
    Pow,
    Symbol,
    adjoint,
    im,
    sympify,
    zoo,
)
from sympy.physics.quantum.state import Bra, Ket
from tqdm import tqdm

from responsefun.AdccProperties import (
    Symmetry,
    build_adcc_properties,
    get_operator_by_name,
)
from responsefun.build_tree import build_tree
from responsefun.IsrFormulation import IsrFormulation, compute_extra_terms
from responsefun.operators import (
    MTM,
    S2S_MTM,
    Moment,
    OneParticleOperator,
    ResponseVector,
    TransitionFrequency,
)
from responsefun.rvec_algebra import bmatrix_vector_product, scalar_product
from responsefun.SumOverStates import SumOverStates
from responsefun.symbols_and_labels import O, gamma

ABC = list(string.ascii_uppercase)


def find_remaining_indices(sos_expr, summation_indices):
    """Find indices of summation of the entered SOS term and return them in a list."""
    assert isinstance(sos_expr, Mul)
    sum_ind = []
    for a in sos_expr.args:
        if isinstance(a, Bra) or isinstance(a, Ket):
            if a.label[0] in summation_indices and a.label[0] not in sum_ind:
                sum_ind.append(a.label[0])
    return sum_ind


def replace_bra_op_ket(expr):
    """Replace Bra(to_state)*op*Ket(from_state) sequence in a SymPy term by an instance of <class
    'responsefun.operators.Moment'>."""
    assert isinstance(expr, Mul)
    subs_dict = {}
    for ia, a in enumerate(expr.args):
        if isinstance(a, OneParticleOperator):
            from_state = expr.args[ia + 1]
            to_state = expr.args[ia - 1]
            key = to_state * a * from_state
            subs_dict[key] = Moment(a.comp, from_state.label[0], to_state.label[0], a.op_type)
    return expr.subs(subs_dict)


def sign_change(no, rvecs_dict, sign=1):
    # TODO: handle this differently, maybe include this already earlier?
    rvec_tup = rvecs_dict[no]
    symmetry = get_operator_by_name(rvec_tup[1]).symmetry
    assert rvec_tup[0] in ["MTM", "S2S_MTM"]
    if symmetry == Symmetry.HERMITIAN:
        pass
    elif symmetry == Symmetry.ANTIHERMITIAN:
        sign *= -1
    else:
        raise NotImplementedError(
            "Only Hermitian and anti-Hermitian operators are implemented."
        )
    if rvec_tup[0] == "S2S_MTM" and rvec_tup[4] == "ResponseVector":
        return sign_change(rvec_tup[5], rvecs_dict, sign)
    assert sign in [1, -1]
    return sign


def _initialize_arguments(
    freqs_in,
    freqs_out,
    damping,
    excited_state,
    extra_terms,
    omegas,  # will be removed
    gamma_val,  # will be removed
    final_state,  # will be removed
):
    external_freqs = []
    if omegas is not None:
        warnings.warn(
            "The omegas keyword is deprecated and will be replaced "
            "by the freqs_in and freqs_out keywords.",
            DeprecationWarning,
        )
        assert freqs_in is None and freqs_out is None
        if isinstance(omegas, tuple):
            external_freqs = [omegas]
        else:
            assert isinstance(omegas, list)
            external_freqs = omegas

    if gamma_val is not None:
        warnings.warn(
            "The gamma_val keyword is deprecated and will be replaced by the damping keyword.",
            DeprecationWarning,
        )
        assert damping is None
        damping = gamma_val
    if final_state is not None:
        warnings.warn(
            "The final_state keyword is deprecated and will be replaced "
            "by the excited_state keyword.",
            DeprecationWarning,
        )
        assert excited_state is None
        assert isinstance(final_state, tuple) and len(final_state) == 2
        excited_state = final_state[1]
    if extra_terms is not True:
        print("Please note that the extra_terms keyword is only intended for testing.")

    if freqs_in is None:
        freqs_in = []
    elif isinstance(freqs_in, tuple):
        freqs_in = [freqs_in]
    else:
        assert isinstance(freqs_in, list)

    if freqs_out is None:
        freqs_out = []
    elif isinstance(freqs_out, tuple):
        freqs_out = [freqs_out]
    else:
        assert isinstance(freqs_out, list)

    external_freqs += freqs_in
    external_freqs += freqs_out

    if damping is None:
        damping = 0.0

    if omegas is None:
        correlation_btw_freq = None
    else:
        correlation_btw_freq = [
            tup for tup in external_freqs if not isinstance(sympify(tup[1]), Number)
        ]
    if excited_state is not None:
        assert isinstance(excited_state, int)

    return (
        freqs_in,
        freqs_out,
        damping,
        excited_state,
        extra_terms,
        external_freqs,  # will be removed
        correlation_btw_freq,  # will be removed
    )


InputSubs = namedtuple("InputSubs", ["all_freqs", "damping", "excited_state"])


def _initialize_sos(
    sos_expr,
    summation_indices,
    freqs_in,
    freqs_out,
    perm_pairs,
    excluded_states,
    symmetric,
    damping,
    excited_state,
    state,
    omegas,  # will be removed
    external_freqs,  # will be removed
    correlation_btw_freq,  # will be removed
):
    sos = SumOverStates(
        sos_expr,
        summation_indices,
        freqs_in=[freq[0] for freq in freqs_in],
        freqs_out=[freq[0] for freq in freqs_out],
        perm_pairs=perm_pairs,
        excluded_states=excluded_states,
        symmetric=symmetric,
        correlation_btw_freq=correlation_btw_freq,
    )
    print(
        "The following SOS expression was entered/generated. "
        f"It consists of {sos.number_of_terms} term(s):\n{sos}\n"
    )
    # check whether the definitions match if frequencies are defined twice
    for freq in external_freqs:
        if sos.correlation_btw_freq:
            same_freq = [sympify(f[1]).subs(sos.correlation_btw_freq) 
                         for f in external_freqs
                         if f[0] == freq[0]]
        else:
            same_freq = [f[1] for f in external_freqs if f[0] == freq[0]]
        assert len(set(same_freq)) == 1

    all_freqs = external_freqs.copy()
    if excited_state is not None:
        all_freqs.append(
            (
                TransitionFrequency(sos.excited_state, real=True),
                state.excitation_energy_uncorrected[excited_state],
            )
        )
        for ies, exstate in enumerate(sos.excluded_states):
            if isinstance(exstate, int) and exstate == excited_state:
                sos.excluded_states[ies] = sos.excited_state
    else:
        assert sos.excited_state is None

    all_freqs_mod = []
    for freq in all_freqs:
        if isinstance(sympify(freq[1]), Number):
            all_freqs_mod.append(freq)
        else:
            all_freqs_mod.append((freq[0], freq[1].subs(all_freqs)))

    all_freqs = all_freqs_mod

    if omegas is None:
        if not sos.check_energy_conservation(all_freqs):
            raise ValueError("Energy conservation check was not passed. See above.")

    input_subs = InputSubs(all_freqs, (gamma, damping), (sos.excited_state, excited_state))
    return sos, input_subs


def determine_rvecs(rvecs_dict_list, input_subs, adcc_prop,
                    state, projection=None, **solver_args):
    matrix = construct_adcmatrix(state.matrix)
    rvecs_dict_tot = {}
    rvecs_solution = {}
    rvecs_mapping = {}
    number_of_unique_rvecs = 0
    print("Solving response equations ...")
    for tup in rvecs_dict_list:
        rvecs_dict = tup[1]
        # check if response equations become equal
        # after inserting values for external_freqs and gamma
        rvecs_dict_mod = {}
        for key, value in rvecs_dict.items():
            om = float(key[2].subs(input_subs.all_freqs))
            gam = float(im(key[3].subs(*input_subs.damping)))
            if gam == 0 and input_subs.damping[1] != 0:
                raise ValueError(
                    "Although the entered SOS expression is real, a value for gamma was specified."
                )
            if key[5] is None:
                new_key = (*key[:2], om, gam, *key[4:])
            else:
                # in case response vectors from the previous iteration have become equal
                new_no = rvecs_mapping[key[5]]
                new_key = (*key[:2], om, gam, key[4], new_no)
            if new_key not in rvecs_dict_mod.keys():
                rvecs_mapping[value] = value
                rvecs_dict_mod[new_key] = value
            else:
                rvecs_mapping[value] = rvecs_dict_mod[new_key]
        number_of_unique_rvecs += len(rvecs_dict_mod)
        # solve response equations
        for key, value in rvecs_dict_mod.items():
            op_type = key[1]
            adcop = adcc_prop[op_type]
            if key[0] == "MTM":
                rhss_shape = np.shape(adcop.integrals)
                response = np.empty(rhss_shape, dtype=object)
                iterables = [list(range(shape)) for shape in rhss_shape]
                components = list(product(*iterables))
                rhs = adcop.modified_transition_moments()
                if key[3] == 0.0:
                    for c in components:
                        # list indices must be integers (1-D operators)
                        c = c[0] if len(c) == 1 else c
                        response[c] = solve_response(
                            matrix, rhs[c], -key[2], gamma=0.0, projection=projection, **solver_args
                        )
                else:
                    for c in components:
                        # list indices must be integers (1-D operators)
                        c = c[0] if len(c) == 1 else c
                        response[c] = solve_response(
                            matrix,
                            RV(rhs[c]),
                            -key[2],
                            gamma=-key[3],
                            projection=projection,
                            **solver_args,
                        )
                rvecs_solution[value] = response
            elif key[0] == "S2S_MTM":
                op_dim = adcop.op_dim
                if key[4] == "ResponseVector":
                    no = key[5]
                    rvecs = rvecs_solution[rvecs_mapping[no]]
                    rhss_shape = (3,) * op_dim + rvecs.shape
                    iterables = [list(range(shape)) for shape in rhss_shape]
                    components = list(product(*iterables))
                    response = np.empty(rhss_shape, dtype=object)
                    for c in components:
                        rvec = rvecs[c[op_dim:]]
                        if isinstance(rvec, AmplitudeVector):
                            bmatrix = adcop.isr_matrix(c[:op_dim])
                            rhs = bmatrix @ rvec
                            if projection is not None:
                                rhs -= projection(rhs)
                            if key[3] == 0.0:
                                response[c] = solve_response(
                                    matrix,
                                    rhs,
                                    -key[2],
                                    gamma=0.0,
                                    projection=projection,
                                    **solver_args,
                                )
                            else:
                                response[c] = solve_response(
                                    matrix,
                                    RV(rhs),
                                    -key[2],
                                    gamma=-key[3],
                                    projection=projection,
                                    **solver_args,
                                )
                        elif isinstance(rvec, RV):
                            bmatrix = adcop.isr_matrix(c[:op_dim])
                            rhs = bmatrix_vector_product(bmatrix, rvec)
                            if projection is not None:
                                raise NotImplementedError(
                                    "Projecting out states from a response equation with a complex "
                                    "right-hand side has not yet been implemented."
                                )
                                # rhs.real -= projection(rhs.real)
                                # rhs.imag -= projection(rhs.imag)
                            if ("solver", "cpp") in list(solver_args.items()):
                                raise NotImplementedError(
                                    "CPP solver only works correctly for purely real rhs."
                                )
                            # TODO: temporary hack --> modify solve_response accordingly
                            rhs = RV(real=rhs.real, imag=-1.0 * rhs.imag)
                            response[c] = solve_response(
                                matrix,
                                rhs,
                                -key[2],
                                gamma=-key[3],
                                projection=projection,
                                **solver_args,
                            )
                        else:
                            raise ValueError()
                    rvecs_solution[value] = response
                elif key[4] == input_subs.excited_state[0]:
                    rhss_shape = (3,) * op_dim
                    iterables = [list(range(shape)) for shape in rhss_shape]
                    components = list(product(*iterables))
                    response = np.empty(rhss_shape, dtype=object)
                    if key[3] == 0.0:
                        for c in components:
                            bmatrix = adcop.isr_matrix(c)
                            rhs = bmatrix @ state.excitation_vector[input_subs.excited_state[1]]
                            if projection is not None:
                                rhs -= projection(rhs)
                            response[c] = solve_response(
                                matrix,
                                rhs,
                                -key[2],
                                gamma=0.0,
                                projection=projection,
                                **solver_args,
                            )
                    else:
                        for c in components:
                            bmatrix = adcop.isr_matrix(c)
                            rhs = bmatrix @ state.excitation_vector[input_subs.excited_state[1]]
                            if projection is not None:
                                rhs -= projection(rhs)
                            response[c] = solve_response(
                                matrix,
                                RV(rhs),
                                -key[2],
                                gamma=-key[3],
                                projection=projection,
                                **solver_args,
                            )
                    rvecs_solution[value] = response
                else:
                    raise ValueError("Unkown response equation.")
            else:
                raise ValueError("Unkown response equation.")
        rvecs_dict_tot.update(dict((value, key) for key, value in rvecs_dict.items()))

    print(
        f"In total, {len(rvecs_dict_tot)} response vectors (with multiple components each) "
        "were defined:"
    )
    for key, value in rvecs_dict_tot.items():
        print(f"X_{{{key}}}: {value}")
    if len(rvecs_dict_tot) > number_of_unique_rvecs:
        print(
            "However, inserting the specified frequency values caused response"
            f" vectors to become equal, so that in the end only {number_of_unique_rvecs}"
            " response vectors had to be determined."
        )
        for lv, rv in rvecs_mapping.items():
            print(f"X_{{{lv}}} = X_{{{rv}}}") if lv != rv else print(f"X_{{{lv}}}")

    return rvecs_dict_tot, rvecs_solution, rvecs_mapping


def process_complex_factor(sos, tensor):
    factor = sos.complex_factor
    print("\nTaking into account imaginary operators:")
    if factor.imag == 0:
        print("The real part of the property is returned.")
        print(f"It was multiplied with a factor of {factor.real}.")
        return factor.real * tensor
    else:
        assert factor.real == 0
        print("The imaginary part of the property is returned.")
        print(f"It was multiplied with a factor of {factor.imag}.")
        return factor.imag * tensor


def evaluate_property_isr(
    state,
    sos_expr,
    summation_indices,
    *,
    perm_pairs=None,
    excluded_states=None,
    freqs_in=None,
    freqs_out=None,
    damping=None,
    excited_state=None,
    symmetric=False,
    extra_terms=True,
    omegas=None,
    gamma_val=None,
    final_state=None,
    **solver_args,
):
    """Compute a molecular property with the ADC/ISR approach from its SOS expression.

    Parameters
    ----------
    state: <class 'adcc.ExcitedStates.ExcitedStates'>
        ExcitedStates object returned by an ADC calculation.

    sos_expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
        SymPy expression of the SOS;
        it can be either the full expression or a single term from which the full expression
        can be generated via permutation.

    summation_indices: list of <class 'sympy.core.symbol.Symbol'>
        List of indices of summation.

    perm_pairs: list of tuples, optional
        List of (op, freq) pairs whose permutation yields the full SOS expression;
        (op, freq): (<class 'responsefun.operators.OneParticleOperator'>,
        <class 'sympy.core.symbol.Symbol'>),
        e.g., [(op_a, -w_o), (op_b, w_1), (op_c, w_2)].

    excluded_states: list of <class 'sympy.core.symbol.Symbol'> or int, optional
        List of states that are excluded from the summation.
        It is important to note that the ground state is represented by the SymPy symbol O,
        while the integer 0 represents the first excited state.
    
    freqs_in: list of tuples, optional
        List of (symbol, value) pairs for the incoming frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'>
        or <class 'sympy.core.mul.Mul'> or <class 'sympy.core.symbol.Symbol'> or float),
        e.g., [(w_1, 0.5), (w_2, 0.5)] or [(w_1, w_f/2), (w_2, w_f/2)].

    freqs_out: list of tuples, optional
        List of (symbol, value) pairs for the outgoing frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'>
        or <class 'sympy.core.mul.Mul'> or <class 'sympy.core.symbol.Symbol'> or float),
        e.g., [(w_o, w_1+w_2)].

    damping: float, optional

    excited_state: int, optional

    symmetric: bool, optional
        Resulting tensor is symmetric;
        by default 'False'.

    extra_terms: bool, optional
        Compute the additional terms that arise when converting the SOS expression to its
        ADC/ISR formulation; should only be used for testing;
        by default 'True'.

    omegas: list of tuples, optional, deprecated
        List of (symbol, value) pairs for the frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'>
        or <class 'sympy.core.symbol.Symbol'> or float),
        e.g., [(w_o, w_1+w_2), (w_1, 0.5), (w_2, 0.5)].

    gamma_val: float, optional, deprecated

    final_state: tuple, optional, deprecated
        (<class 'sympy.core.symbol.Symbol'>, int), e.g., (f, 0).

    Returns
    ----------
    <class 'numpy.ndarray'>
        Resulting tensor with components ABC....
    """
    (
        freqs_in,
        freqs_out,
        damping,
        excited_state,
        extra_terms,
        external_freqs,
        correlation_btw_freq,
    ) = _initialize_arguments(
        freqs_in,
        freqs_out,
        damping,
        excited_state,
        extra_terms,
        omegas,
        gamma_val,
        final_state,
    )

    sos, input_subs = _initialize_sos(
        sos_expr,
        summation_indices,
        freqs_in,
        freqs_out,
        perm_pairs,
        excluded_states,
        symmetric,
        damping,
        excited_state,
        state,
        omegas,
        external_freqs,
        correlation_btw_freq,
    )

    isr = IsrFormulation(sos, extra_terms, print_extra_term_dict=True)
    print(
        f"The SOS expression was transformed into the following ADC/ISR formulation:\n{isr}\nThus, "
        f"{isr.number_of_extra_terms} non-vanishing terms were identified that must be additionally"
        " considered due to the definition of the ADC matrices.\n"
    )
    print("Building tree to determine suitable response vectors ...")
    rvecs_dict_list = build_tree(isr.mod_expr)

    # prepare the projection of the states excluded from the summation
    to_be_projected_out = []
    for exstate in sos.excluded_states:
        if exstate == O:
            continue  # the ADC quantities do not include the ground state anyway
        elif isinstance(exstate, int):
            to_be_projected_out.append(exstate)
        else:
            assert input_subs.excited_state[0] is not None
            assert exstate == input_subs.excited_state[0]
            to_be_projected_out.append(input_subs.excited_state[1])
    if to_be_projected_out:
        print(
            f"The following states are projected out from the ADC matrices: {to_be_projected_out}"
        )
        if len(to_be_projected_out) != 1:
            raise NotImplementedError("It is not yet possible to project out more than one state.")
        exstate = to_be_projected_out[0]
        v_f = state.excitation_vector[exstate]

        def projection(X, bl=None):
            if bl:
                vb = getattr(v_f, bl)
                return vb * (vb.dot(X)) / (vb.dot(vb))
            else:
                return v_f * (v_f @ X) / (v_f @ v_f)

    else:
        projection = None

    # store adcc properties for the required operators in a dict
    adcc_prop = {}
    for op_type in sos.operator_types:
        adcc_prop[op_type] = build_adcc_properties(state, op_type)

    rvecs_dict_tot, rvecs_solution, rvecs_mapping = determine_rvecs(
        rvecs_dict_list, input_subs, adcc_prop, state, projection, **solver_args
    )
    if rvecs_dict_list:
        root_expr = rvecs_dict_list[-1][0]
    else:
        root_expr = isr.mod_expr

    dtype = float
    if input_subs.damping[1] != 0.0:
        dtype = complex
    res_tens = np.zeros((3,) * sos.order, dtype=dtype)

    if isinstance(root_expr, Add):
        term_list = [arg for arg in root_expr.args]
    else:
        term_list = [root_expr]

    if sos.symmetric:
        components = list(
            combinations_with_replacement([0, 1, 2], sos.order)
        )  # if tensor is symmetric
    else:
        components = list(product([0, 1, 2], repeat=sos.order))
    for c in components:
        comp_map = {ABC[ic]: cc for ic, cc in enumerate(c)}

        for term in term_list:
            subs_dict = dict(input_subs.all_freqs)
            subs_dict[input_subs.damping[0]] = input_subs.damping[1]
            for i, a in enumerate(term.args):
                oper_a = a
                if isinstance(a, adjoint):
                    oper_a = a.args[0]
                if isinstance(oper_a, ResponseVector) and oper_a == a:  # vec * X
                    comps_right_v = tuple([comp_map[char] for char in list(oper_a.comp)])
                    right_v = rvecs_solution[rvecs_mapping[oper_a.no]][comps_right_v]

                    lhs = term.args[i - 1]
                    if isinstance(lhs, S2S_MTM):  # vec * B * X --> transition polarizability
                        adcop = adcc_prop[lhs.op_type]
                        lhs2 = term.args[i - 2]
                        key = lhs2 * lhs * a
                        if isinstance(lhs2, adjoint) and \
                            isinstance(lhs2.args[0], ResponseVector):  # Dagger(X) * B * X
                            comps_left_v = tuple(
                                [comp_map[char] for char in list(lhs2.args[0].comp)]
                            )
                            sign = sign_change(lhs2.args[0].no, rvecs_dict_tot)
                            left_v = (
                                sign * rvecs_solution[rvecs_mapping[lhs2.args[0].no]][comps_left_v]
                            )
                        elif isinstance(lhs2, Bra):  # <f| * B * X
                            assert lhs2.label[0] == input_subs.excited_state[0]
                            left_v = state.excitation_vector[input_subs.excited_state[1]]
                        else:
                            raise ValueError("Expression cannot be evaluated.")
                        comps_dip = tuple([comp_map[char] for char in list(lhs.comp)])
                        subs_dict[key] = adcop.transition_polarizability(left_v, right_v, comps_dip)
                    elif isinstance(lhs, adjoint) and isinstance(lhs.args[0], MTM):  # Dagger(F) * X
                        adcop = adcc_prop[lhs.args[0].op_type]
                        comps_left_v = tuple([comp_map[char] for char in list(lhs.args[0].comp)])
                        # list indices must be integers (1-D operators)
                        comps_left_v = comps_left_v[0] if len(comps_left_v) == 1 else comps_left_v
                        left_v = adcop.modified_transition_moments_reverse(comps_left_v)
                        subs_dict[lhs * a] = scalar_product(left_v, right_v)
                    elif isinstance(lhs, adjoint) and \
                        isinstance(lhs.args[0], ResponseVector):  # Dagger(X) * X
                        comps_left_v = tuple([comp_map[char] for char in list(lhs.args[0].comp)])
                        sign = sign_change(lhs.args[0].no, rvecs_dict_tot)
                        left_v = sign * rvecs_solution[rvecs_mapping[lhs.args[0].no]][comps_left_v]
                        subs_dict[lhs * a] = scalar_product(left_v, right_v)
                    else:
                        raise ValueError("Expression cannot be evaluated.")
                elif isinstance(oper_a, ResponseVector) and oper_a != a:  # Dagger(X) * vec
                    rhs = term.args[i + 1]
                    comps_left_v = tuple([comp_map[char] for char in list(oper_a.comp)])
                    sign = sign_change(oper_a.no, rvecs_dict_tot)
                    left_v = sign * rvecs_solution[rvecs_mapping[oper_a.no]][comps_left_v]

                    if isinstance(
                        rhs, S2S_MTM
                    ):  # Dagger(X) * B * vec --> transition polarizability
                        adcop = adcc_prop[rhs.op_type]
                        rhs2 = term.args[i + 2]
                        key = a * rhs * rhs2
                        if isinstance(
                            rhs2, ResponseVector
                        ):  # Dagger(X) * B * X (taken care of above)
                            continue
                        elif isinstance(rhs2, Ket):  # Dagger(X) * B * |f>
                            assert rhs2.label[0] == input_subs.excited_state[0]
                            right_v = state.excitation_vector[input_subs.excited_state[1]]
                        else:
                            raise ValueError("Expression cannot be evaluated.")
                        comps_dip = tuple([comp_map[char] for char in list(rhs.comp)])
                        subs_dict[key] = adcop.transition_polarizability(left_v, right_v, comps_dip)
                    elif isinstance(rhs, MTM):  # Dagger(X) * F
                        adcop = adcc_prop[rhs.op_type]
                        comps_right_v = tuple([comp_map[char] for char in list(rhs.comp)])
                        # list indices must be integers (1-D operators)
                        comps_right_v = (
                            comps_right_v[0] if len(comps_right_v) == 1 else comps_right_v
                        )
                        right_v = adcop.modified_transition_moments(comps_right_v)
                        subs_dict[a * rhs] = scalar_product(left_v, right_v)
                    elif isinstance(rhs, ResponseVector):  # Dagger(X) * X (taken care of above)
                        continue
                    else:
                        raise ValueError("Expression cannot be evaluated.")

                elif isinstance(a, Moment):
                    adcop = adcc_prop[a.op_type]
                    comps_dipmom = tuple([comp_map[char] for char in list(a.comp)])
                    if a.from_state == O and a.to_state == O:
                        gs_moment = adcop.gs_moment
                        subs_dict[a] = gs_moment[comps_dipmom]
                    elif a.from_state == O and a.to_state == input_subs.excited_state[0]:
                        tdms = adcop.transition_moment
                        subs_dict[a] = tdms[input_subs.excited_state[1]][comps_dipmom]
                    else:
                        raise ValueError("Unknown transition moment.")
            res = term.subs(subs_dict)
            if res == zoo:
                raise ZeroDivisionError()
            res_tens[c] += res

        if sos.symmetric:
            perms = list(permutations(c))  # if tensor is symmetric
            for pe in perms:
                res_tens[pe] = res_tens[c]
    res_tens = process_complex_factor(sos, res_tens)
    print("========== The requested tensor was formed. ==========")
    return res_tens


def evaluate_property_sos(
    state,
    sos_expr,
    summation_indices,
    *,
    perm_pairs=None,
    excluded_states=None,
    freqs_in=None,
    freqs_out=None,
    damping=None,
    excited_state=None,
    symmetric=False,
    extra_terms=True,
    omegas=None,
    gamma_val=None,
    final_state=None,
):
    """Compute a molecular property from its SOS expression.

    Parameters
    ----------
    state: <class 'adcc.ExcitedStates.ExcitedStates'>
        ExcitedStates object returned by an ADC calculation that includes all states of the system.

    sos_expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
        SymPy expression of the SOS;
        it can be either the full expression or a single term from which the full expression
        can be generated via permutation.

    summation_indices: list of <class 'sympy.core.symbol.Symbol'>
        List of indices of summation.

    perm_pairs: list of tuples, optional
        List of (op, freq) pairs whose permutation yields the full SOS expression;
        (op, freq): (<class 'responsefun.operators.OneParticleOperator'>,
        <class 'sympy.core.symbol.Symbol'>),
        e.g., [(op_a, -w_o), (op_b, w_1), (op_c, w_2)].

    excluded_states: list of <class 'sympy.core.symbol.Symbol'> or int, optional
        List of states that are excluded from the summation.
        It is important to note that the ground state is represented by the SymPy symbol O,
        while the integer 0 represents the first excited state.
    
    freqs_in: list of tuples, optional
        List of (symbol, value) pairs for the incoming frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'>
        or <class 'sympy.core.mul.Mul'> or <class 'sympy.core.symbol.Symbol'> or float),
        e.g., [(w_1, 0.5), (w_2, 0.5)] or [(w_1, w_f/2), (w_2, w_f/2)].

    freqs_out: list of tuples, optional
        List of (symbol, value) pairs for the outgoing frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'>
        or <class 'sympy.core.mul.Mul'> or <class 'sympy.core.symbol.Symbol'> or float),
        e.g., [(w_o, w_1+w_2)].

    damping: float, optional

    excited_state: int, optional

    symmetric: bool, optional
        Resulting tensor is symmetric;
        by default 'False'.

    extra_terms: bool, optional
        Compute the additional terms that arise when converting the SOS expression to its
        ADC/ISR formulation; should only be used for testing;
        by default 'True'.

    omegas: list of tuples, optional, deprecated
        List of (symbol, value) pairs for the frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'>
        or <class 'sympy.core.symbol.Symbol'> or float),
        e.g., [(w_o, w_1+w_2), (w_1, 0.5), (w_2, 0.5)].

    gamma_val: float, optional, deprecated

    final_state: tuple, optional, deprecated
        (<class 'sympy.core.symbol.Symbol'>, int), e.g., (f, 0).

    Returns
    ----------
    <class 'numpy.ndarray'>
        Resulting tensor with components ABC....
    """
    (
        freqs_in,
        freqs_out,
        damping,
        excited_state,
        extra_terms,
        external_freqs,
        correlation_btw_freq,
    ) = _initialize_arguments(
        freqs_in,
        freqs_out,
        damping,
        excited_state,
        extra_terms,
        omegas,
        gamma_val,
        final_state,
    )

    sos, input_subs = _initialize_sos(
        sos_expr,
        summation_indices,
        freqs_in,
        freqs_out,
        perm_pairs,
        excluded_states,
        symmetric,
        damping,
        excited_state,
        state,
        omegas,  # will be removed
        external_freqs,  # will be removed
        correlation_btw_freq,  # will be removed
    )

    # all terms are stored as dictionaries in a list
    if isinstance(sos.expr, Add):
        term_list = [
            {
                "expr": term,
                "summation_indices": sos.summation_indices,
                "transition_frequencies": sos.transition_frequencies,
            }
            for term in sos.expr.args
        ]
    else:
        term_list = [
            {
                "expr": sos.expr,
                "summation_indices": sos.summation_indices,
                "transition_frequencies": sos.transition_frequencies,
            }
        ]
    if extra_terms:
        print("Determining extra terms ...")
        ets = compute_extra_terms(
            sos.expr,
            sos.summation_indices,
            excluded_states=sos.excluded_states,
            correlation_btw_freq=sos.correlation_btw_freq,
            print_extra_term_dict=True,
        )
        if isinstance(ets, Add):
            et_list = list(ets.args)
        elif isinstance(ets, Mul):
            et_list = [ets]
        else:
            et_list = []
        print(
            f"{len(et_list)} non-vanishing terms were identified that must be additionally "
            "considered due to the definition of the adcc properties.\n"
        )
        for et in et_list:
            # the extra terms contain less indices of summation
            sum_ind = find_remaining_indices(et, sos.summation_indices)
            trans_freq = [TransitionFrequency(index, real=True) for index in sum_ind]
            term_list.append(
                {"expr": et, "summation_indices": sum_ind, "transition_frequencies": trans_freq}
            )

    if excited_state:
        for ies, exstate in enumerate(sos.excluded_states):
            if exstate == excited_state[0]:
                sos.excluded_states[ies] = excited_state[1]

    dtype = float
    if damping != 0.0:
        dtype = complex
    res_tens = np.zeros((3,) * sos.order, dtype=dtype)

    if sos.symmetric:
        components = list(
            combinations_with_replacement([0, 1, 2], sos.order)
        )  # if tensor is symmetric
    else:
        components = list(product([0, 1, 2], repeat=sos.order))

    # store adcc properties for the required operators in a dict
    adcc_prop = {}
    for op_type in sos.operator_types:
        adcc_prop[op_type] = build_adcc_properties(state, op_type)

    print(f"Summing over {len(state.excitation_energy_uncorrected)} excited states ...")
    for term_dict in tqdm(term_list):
        mod_expr = replace_bra_op_ket(term_dict["expr"].subs(sos.correlation_btw_freq))
        sum_ind = term_dict["summation_indices"]

        # values that the indices of summation can take on
        indices = list(
            product(range(len(state.excitation_energy_uncorrected)), repeat=len(sum_ind))
        )
        dip_mom_list = [a for a in mod_expr.args if isinstance(a, Moment)]
        for i in indices:
            state_map = {sum_ind[ii]: ind for ii, ind in enumerate(i)}

            # skip the rest of the loop for this iteration if it corresponds to one
            # of the excluded states
            if set(sos.excluded_states).intersection(set(state_map.values())):
                continue

            if input_subs.excited_state[0] is not None:
                state_map[input_subs.excited_state[0]] = input_subs.excited_state[1]
            for c in components:
                comp_map = {ABC[ic]: cc for ic, cc in enumerate(c)}
                subs_dict = dict(input_subs.all_freqs)
                subs_dict[input_subs.damping[0]] = input_subs.damping[1]

                for si, tf in zip(sum_ind, term_dict["transition_frequencies"]):
                    subs_dict[tf] = state.excitation_energy_uncorrected[state_map[si]]

                for a in dip_mom_list:
                    adcop = adcc_prop[a.op_type]
                    comps_dipmom = tuple([comp_map[char] for char in list(a.comp)])
                    if a.from_state == O and a.to_state == O:
                        gs_moment = adcop.gs_moment
                        subs_dict[a] = gs_moment[comps_dipmom]
                    elif a.from_state == O:
                        index = state_map[a.to_state]
                        tdms = adcop.transition_moment
                        subs_dict[a] = tdms[index][comps_dipmom]
                    elif a.to_state == O:
                        index = state_map[a.from_state]
                        tdms = adcop.transition_moment_reverse
                        subs_dict[a] = tdms[index][comps_dipmom]
                    else:
                        index1 = state_map[a.from_state]
                        index2 = state_map[a.to_state]
                        if a.from_state in sum_ind and a.to_state in sum_ind:  # e.g., <n|op|m>
                            s2s_tdms = adcop.state_to_state_transition_moment
                            subs_dict[a] = s2s_tdms[index1, index2][comps_dipmom]
                        elif a.from_state in sum_ind:  # e.g., <f|op|n>
                            s2s_tdms_f = adcop.s2s_tm_view(final_state=index2)
                            subs_dict[a] = s2s_tdms_f[index1][comps_dipmom]
                        elif a.to_state in sum_ind:  # e.g., <n|op|f>
                            s2s_tdms_f = adcop.s2s_tm_view(initial_state=index1)
                            subs_dict[a] = s2s_tdms_f[index2][comps_dipmom]
                        else:
                            raise ValueError()
                res = mod_expr.xreplace(subs_dict)
                if res == zoo:
                    raise ZeroDivisionError()
                res_tens[c] += res
                if sos.symmetric:
                    perms = list(permutations(c))  # if tensor is symmetric
                    for pe in perms:
                        res_tens[pe] = res_tens[c]
    res_tens = process_complex_factor(sos, res_tens)
    print("========== The requested tensor was formed. ==========")
    return res_tens


def evaluate_property_sos_fast(
    state,
    sos_expr,
    summation_indices,
    *,
    perm_pairs=None,
    excluded_states=None,
    freqs_in=None,
    freqs_out=None,
    damping=None,
    excited_state=None,
    extra_terms=True,
    omegas=None,
    gamma_val=None,
    final_state=None,
):
    """Compute a molecular property from its SOS expression using the Einstein summation convention.

    Parameters
    ----------
    state: <class 'adcc.ExcitedStates.ExcitedStates'>
        ExcitedStates object returned by an ADC calculation that includes all states of the system.

    sos_expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
        SymPy expression of the SOS;
        it can be either the full expression or a single term from which the full expression
        can be generated via permutation. It already includes the additional terms.

    summation_indices: list of <class 'sympy.core.symbol.Symbol'>
        List of indices of summation.

    perm_pairs: list of tuples, optional
        List of (op, freq) pairs whose permutation yields the full SOS expression;
        (op, freq): (<class 'responsefun.operators.OneParticleOperator'>,
        <class 'sympy.core.symbol.Symbol'>),
        e.g., [(op_a, -w_o), (op_b, w_1), (op_c, w_2)].

    excluded_states: list of <class 'sympy.core.symbol.Symbol'> or int, optional
        List of states that are excluded from the summation.
        It is important to note that the ground state is represented by the SymPy symbol O,
        while the integer 0 represents the first excited state.
    
    freqs_in: list of tuples, optional
        List of (symbol, value) pairs for the incoming frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'>
        or <class 'sympy.core.mul.Mul'> or <class 'sympy.core.symbol.Symbol'> or float),
        e.g., [(w_1, 0.5), (w_2, 0.5)] or [(w_1, w_f/2), (w_2, w_f/2)].

    freqs_out: list of tuples, optional
        List of (symbol, value) pairs for the outgoing frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'>
        or <class 'sympy.core.mul.Mul'> or <class 'sympy.core.symbol.Symbol'> or float),
        e.g., [(w_o, w_1+w_2)].

    damping: float, optional

    excited_state: int, optional

    extra_terms: bool, optional
        Compute the additional terms that arise when converting the SOS expression to its
        ADC/ISR formulation; should only be used for testing;
        by default 'True'.

    omegas: list of tuples, optional, deprecated
        List of (symbol, value) pairs for the frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'>
        or <class 'sympy.core.symbol.Symbol'> or float),
        e.g., [(w_o, w_1+w_2), (w_1, 0.5), (w_2, 0.5)].

    gamma_val: float, optional, deprecated

    final_state: tuple, optional, deprecated
        (<class 'sympy.core.symbol.Symbol'>, int), e.g., (f, 0).

    Returns
    ----------
    <class 'numpy.ndarray'>
        Resulting tensor with components ABC....
    """
    (
        freqs_in,
        freqs_out,
        damping,
        excited_state,
        extra_terms,
        external_freqs,
        correlation_btw_freq,
    ) = _initialize_arguments(
        freqs_in,
        freqs_out,
        damping,
        excited_state,
        extra_terms,
        omegas,
        gamma_val,
        final_state,
    )

    sos, input_subs = _initialize_sos(
        sos_expr,
        summation_indices,
        freqs_in,
        freqs_out,
        perm_pairs,
        excluded_states,
        False,
        damping,
        excited_state,
        state,
        omegas,
        external_freqs,
        correlation_btw_freq,
    )

    subs_dict = dict(input_subs.all_freqs)
    subs_dict[input_subs.damping[0]] = input_subs.damping[1]

    if extra_terms:
        print("Determining extra terms ...")
        computed_terms = compute_extra_terms(
            sos.expr,
            sos.summation_indices,
            excluded_states=sos.excluded_states,
            correlation_btw_freq=sos.correlation_btw_freq,
            print_extra_term_dict=True,
        )
        if computed_terms == 0:
            number_of_extra_terms = 0
        elif isinstance(computed_terms, Add):
            number_of_extra_terms = len(computed_terms.args)
        else:
            number_of_extra_terms = 1
        print(
            f"{number_of_extra_terms} non-vanishing terms were identified that must be "
            "additionally considered due to the definition of the adcc properties.\n"
        )
        sos_with_et = sos.expr + computed_terms
        sos_expr_mod = sos_with_et.subs(sos.correlation_btw_freq)
    else:
        sos_expr_mod = sos.expr.subs(sos.correlation_btw_freq)

    dtype = float
    if input_subs.damping[1] != 0.0:
        dtype = complex
    res_tens = np.zeros((3,) * sos.order, dtype=dtype)

    if isinstance(sos_expr_mod, Add):
        term_list = [replace_bra_op_ket(arg) for arg in sos_expr_mod.args]
    else:
        term_list = [replace_bra_op_ket(sos_expr_mod)]
    print(
        f"Summing over {len(state.excitation_energy_uncorrected)} excited states "
        "using the Einstein summation convention ..."
    )

    # store adcc properties for the required operators in a dict
    adcc_prop = {}
    for op_type in sos.operator_types:
        adcc_prop[op_type] = build_adcc_properties(state, op_type)

    for it, term in enumerate(term_list):
        einsum_list = []
        factor = 1
        divergences = []
        for a in term.args:
            if isinstance(a, Moment):
                adcop = adcc_prop[a.op_type]
                if a.from_state == O and a.to_state == O:  # <0|op|0>
                    gs_moment = adcop.gs_moment
                    einsum_list.append(("", a.comp, gs_moment))
                elif a.from_state == O:
                    tdms = adcop.transition_moment  # TODO: correct sign?
                    if a.to_state in sos.summation_indices:  # e.g., <n|op|0>
                        einsum_list.append((str(a.to_state), a.comp, tdms))
                    else:  # e.g., <f|op|0>
                        einsum_list.append(("", a.comp, tdms[input_subs.excited_state[1]]))
                elif a.to_state == O:
                    tdms = adcop.transition_moment_reverse
                    if a.from_state in sos.summation_indices:  # e.g., <0|op|n>
                        einsum_list.append((str(a.from_state), a.comp, tdms))
                    else:  # e.g., <0|op|f>
                        einsum_list.append(("", a.comp, tdms[input_subs.excited_state[1]]))
                else:
                    if (
                        a.from_state in sos.summation_indices
                        and a.to_state in sos.summation_indices
                    ):  # e.g., <n|op|m>
                        s2s_tdms = adcop.state_to_state_transition_moment
                        einsum_list.append((str(a.from_state) + str(a.to_state), a.comp, s2s_tdms))
                    elif (
                        a.from_state in sos.summation_indices and \
                            a.to_state == input_subs.excited_state[0]
                    ):  # e.g., <f|op|n>
                        s2s_tdms_f = adcop.s2s_tm_view(final_state=input_subs.excited_state[1])
                        einsum_list.append((str(a.from_state), a.comp, s2s_tdms_f))
                    elif (
                        a.to_state in sos.summation_indices and \
                            a.from_state == input_subs.excited_state[0]
                    ):  # e.g., <n|op|f>
                        s2s_tdms_f = adcop.s2s_tm_view(initial_state=input_subs.excited_state[1])
                        einsum_list.append((str(a.to_state), a.comp, s2s_tdms_f))
                    else:
                        raise ValueError()

            elif isinstance(a, Pow):
                pow_expr = a.args[0].subs(subs_dict)
                if pow_expr == 0:
                    raise ZeroDivisionError()
                index = None
                shift = 0
                if isinstance(pow_expr, Add):
                    pow_expr_list = [arg for arg in pow_expr.args]
                else:
                    pow_expr_list = [pow_expr]
                for aa in pow_expr_list:
                    if aa in sos.transition_frequencies:
                        index = aa.state
                    elif isinstance(aa, Float) or isinstance(aa, Integer):
                        # convert SymPy object to float
                        shift += float(aa)
                    elif isinstance(aa, Mul) and aa.args[1] is I:
                        shift += 1j * float(aa.args[0])
                    else:
                        raise ValueError()
                if index is None:
                    if shift:
                        einsum_list.append(("", "", 1 / (shift)))
                    else:
                        raise ZeroDivisionError()
                else:
                    array = 1 / (state.excitation_energy_uncorrected + shift)
                    if np.inf in array:
                        index_with_inf = np.where(array == np.inf)
                        assert len(index_with_inf) == 1
                        assert len(index_with_inf[0]) == 1
                        divergences.append((index, index_with_inf[0][0]))
                    einsum_list.append((str(index), "", array))

            elif isinstance(a, Integer) or isinstance(a, Float):
                factor *= float(a)

            else:
                raise TypeError(f"The following type was not recognized: {type(a)}.")

        if len(divergences) != 0:
            print(
                "The following divergences have been found (explaining the RuntimeWarning): ",
                divergences,
            )
        einsum_left = ""
        einsum_right = ""
        array_list = []
        removed_divergences = []
        # create string of subscript labels and list of np.arrays for np.einsum
        for tup in einsum_list:
            state_str, comp_str, array = tup
            einsum_left += state_str + comp_str + ","
            einsum_right += comp_str
            # remove excluded states from corresponding arrays
            if state_str:
                for exstate in sos.excluded_states:
                    if exstate == O:
                        continue
                    if isinstance(exstate, int):
                        index_to_delete = exstate
                    else:
                        assert input_subs.excited_state[0] is not None
                        assert exstate == input_subs.excited_state[0]
                        index_to_delete = input_subs.excited_state[1]
                    for axis in range(len(state_str)):
                        array = np.delete(array, index_to_delete, axis=axis)
                        removed_divergences.append(
                            (Symbol(state_str[axis], real=True), index_to_delete)
                        )
            array_list.append(array)
        removed_divergences = list(set(removed_divergences))
        divergences_copied = divergences.copy()
        for rd in removed_divergences:
            if rd in divergences:
                divergences_copied.remove(rd)
        if len(divergences) != 0:
            if len(divergences_copied) != 0:
                raise ZeroDivisionError(
                    "Not all divergences that occured could be eliminated."
                    f"The following divergences remain: {divergences}."
                )
            else:
                print("However, all of these divergences have been successfully removed.")
        einsum_left_mod = einsum_left[:-1]
        einsum_right_list = list(set(einsum_right))
        einsum_right_list.sort()
        einsum_right_mod = "".join(einsum_right_list)
        einsum_string = einsum_left_mod + " -> " + einsum_right_mod
        print(
            f"Created string of subscript labels that is used by np.einsum for term {it+1}:\n",
            einsum_string,
        )
        res_tens += factor * np.einsum(einsum_string, *array_list)

    res_tens = process_complex_factor(sos, res_tens)
    print("========== The requested tensor was formed. ==========")
    return res_tens
