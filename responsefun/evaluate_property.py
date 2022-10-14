import numpy as np
import string
from scipy.constants import physical_constants

from sympy.physics.quantum.state import Bra, Ket
from sympy import Symbol, Mul, Add, Pow, adjoint, im, Float, Integer, zoo, I
from itertools import permutations, product, combinations_with_replacement

from responsefun.symbols_and_labels import gamma, O
from responsefun.response_operators import (
        MTM, S2S_MTM, ResponseVector, DipoleOperator,
        DipoleMoment, TransitionFrequency, LeviCivita
)
from responsefun.sum_over_states import TransitionMoment, SumOverStates
from responsefun.isr_conversion import IsrFormulation, compute_extra_terms
from responsefun.build_tree import build_tree
from responsefun.bmatrix_vector_product import bmatrix_vector_product
from responsefun.adcc_properties import AdccProperties, available_operators

from adcc import AmplitudeVector
from adcc.workflow import construct_adcmatrix
from respondo.solve_response import solve_response, transition_polarizability, transition_polarizability_complex
from respondo.cpp_algebra import ResponseVector as RV
from tqdm import tqdm


Hartree = physical_constants["hartree-electron volt relationship"][0]
ABC = list(string.ascii_uppercase)

# Levi-Civita tensor
lc_tensor = np.zeros((3, 3, 3))
lc_tensor[0, 1, 2] = lc_tensor[1, 2, 0] = lc_tensor[2, 0, 1] = 1
lc_tensor[2, 1, 0] = lc_tensor[0, 2, 1] = lc_tensor[1, 0, 2] = -1


def _check_omegas_and_final_state(sos_expr, omegas, correlation_btw_freq, gamma_val, final_state):
    """Checks for errors in the entered frequencies or the final state.
    """
    if isinstance(sos_expr, Add):
        arg_list = [a for term in sos_expr.args for a in term.args]
        denom_list = [a.args[0] for a in arg_list if isinstance(a, Pow)]
    else:
        arg_list = [a for a in sos_expr.args]
        denom_list = [a.args[0] for a in arg_list if isinstance(a, Pow)]

    if omegas:
        omega_symbols = [tup[0] for tup in omegas]
        for o in omega_symbols:
            if omega_symbols.count(o) > 1:
                pass
                # raise ValueError("Two different values were given for the same frequency.")

        sum_freq = [freq for tup in correlation_btw_freq for freq in tup[1].args]
        check_dict = {o[0]: False for o in omegas}
        for o in check_dict:
            for denom in denom_list:
                if o in denom.args or -o in denom.args or o in sum_freq or -o in sum_freq:
                    check_dict[o] = True
                    break
        if False in check_dict.values():
            pass
            # raise ValueError(
            #         "A frequency was specified that is not included in"
            #         "the entered SOS expression.\nomegas: {}".format(check_dict)
            # )

    if gamma_val:
        for denom in denom_list:
            if 1.0*gamma*I not in denom.args and -1.0*gamma*I not in denom.args:
                raise ValueError("Although the entered SOS expression is real, a value for gamma was specified.")

    if final_state:
        check_f = False
        for a in arg_list:
            if a == Bra(final_state[0]) or a == Ket(final_state[0]):
                check_f = True
                break
        if not check_f:
            raise ValueError("A final state was mistakenly specified.")


def find_remaining_indices(sos_expr, summation_indices):
    """Find indices of summation of the entered SOS term and return them in a list.
    """
    assert isinstance(sos_expr, Mul)
    sum_ind = []
    for a in sos_expr.args:
        if isinstance(a, Bra) or isinstance(a, Ket):
            if a.label[0] in summation_indices and a.label[0] not in sum_ind:
                sum_ind.append(a.label[0])
    return sum_ind


def replace_bra_op_ket(expr):
    """Replace Bra(to_state)*op*Ket(from_state) sequence in a SymPy term
    by an instance of <class 'responsetree.response_operators.DipoleMoment'>.
    """
    assert type(expr) == Mul
    subs_dict = {}
    for ia, a in enumerate(expr.args):
        if isinstance(a, DipoleOperator):
            from_state = expr.args[ia+1]
            to_state = expr.args[ia-1]
            key = to_state*a*from_state
            subs_dict[key] = DipoleMoment(a.comp, from_state.label[0], to_state.label[0], a.op_type)
    return expr.subs(subs_dict)


def scalar_product(left_v, right_v):
    """Evaluate the scalar product between two instances of ResponseVector and/or AmplitudeVector."""
    if isinstance(left_v, AmplitudeVector):
        lv = RV(left_v)
    else:
        lv = left_v.copy()
    if isinstance(right_v, AmplitudeVector):
        rv = RV(right_v)
    else:
        rv = right_v.copy()
    assert isinstance(lv, RV) and isinstance(rv, RV)
    real = lv.real @ rv.real - lv.imag @ rv.imag
    imag = lv.real @ rv.imag + lv.imag @ rv.real
    if imag == 0:
        return real
    else:
        return real + 1j*imag


def sign_change(no, rvecs_dict, sign=1):
    rvec_tup = rvecs_dict[no]
    symmetry = available_operators[rvec_tup[1]][1]
    if rvec_tup[0] == "MTM":
        if symmetry == 1:  # Hermitian operators
            pass
        elif symmetry == 2:  # anti-Hermitian operators
            sign *= -1
        else:
            raise NotImplementedError("Only Hermitian and anti-Hermitian operators are implemented.")
    elif rvec_tup[0] == "S2S_MTM":
        if symmetry == 1:  # Hermitian operators
            pass
        elif symmetry == 2:  # anti-Hermitian operators
            sign *= -1
        else:
            raise NotImplementedError("Only Hermitian and anti-Hermitian operators are implemented.")
        if rvec_tup[4] == "ResponseVector":
            return sign_change(rvec_tup[5], rvecs_dict, sign)
    else:
        raise ValueError()

    # return True if the sign must be changed
    if sign == 1:
        return False
    elif sign == -1:
        return True
    else:
        raise ValueError()


def evaluate_property_isr(
        state, sos_expr, summation_indices, omegas=None, gamma_val=0.0,
        final_state=None, perm_pairs=None, extra_terms=True, symmetric=False, excluded_states=None, **solver_args):
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

    omegas: list of tuples, optional
        List of (symbol, value) pairs for the frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'>
        or <class 'sympy.core.symbol.Symbol'> or float),
        e.g., [(w_o, w_1+w_2), (w_1, 0.5), (w_2, 0.5)].

    gamma_val: float, optional

    final_state: tuple, optional
        (<class 'sympy.core.symbol.Symbol'>, int), e.g., (f, 0).

    perm_pairs: list of tuples, optional
        List of (op, freq) pairs whose permutation yields the full SOS expression;
        (op, freq): (<class 'responsetree.response_operators.DipoleOperator'>, <class 'sympy.core.symbol.Symbol'>),
        e.g., [(op_a, -w_o), (op_b, w_1), (op_c, w_2)].

    extra_terms: bool, optional
        Compute the additional terms that arise when converting the SOS expression to its ADC/ISR formulation;
        by default 'True'.

    symmetric: bool, optional
        Resulting tensor is symmetric;
        by default 'False'.

    excluded_states: list of <class 'sympy.core.symbol.Symbol'> or int, optional
        List of states that are excluded from the summation.
        It is important to note that the ground state is represented by the SymPy symbol O, while the integer 0
        represents the first excited state.

    Returns
    ----------
    <class 'numpy.ndarray'>
        Resulting tensor.
    """
    matrix = construct_adcmatrix(state.matrix)
    property_method = state.property_method
    mp = matrix.ground_state

    if omegas is None:
        omegas = []
    elif isinstance(omegas, tuple):
        omegas = [omegas]
    else:
        assert isinstance(omegas, list)
    assert isinstance(symmetric, bool)

    # create SumOverStates object from input
    correlation_btw_freq = [tup for tup in omegas if type(tup[1]) == Symbol or type(tup[1]) == Add]
    sos = SumOverStates(
            sos_expr, summation_indices, correlation_btw_freq=correlation_btw_freq,
            perm_pairs=perm_pairs, excluded_states=excluded_states
    )
    print(
        f"\nThe following SOS expression was entered/generated. It consists of {sos.number_of_terms} term(s):\n{sos}\n"
    )

    # store adcc properties for the required operators in a dict
    adcc_prop_dict = {}
    for op_type in sos.operator_types:
        adcc_prop_dict[op_type] = AdccProperties(state, op_type)

    all_omegas = omegas.copy()
    if final_state:
        assert type(final_state) == tuple and len(final_state) == 2
        all_omegas.append(
                (TransitionFrequency(final_state[0], real=True),
                 state.excitation_energy_uncorrected[final_state[1]])
        )
        for ies, exstate in enumerate(sos.excluded_states):
            if isinstance(exstate, int) and exstate == final_state[1]:
                sos.excluded_states[ies] = final_state[0]
    else:
        assert final_state is None

    _check_omegas_and_final_state(sos.expr, omegas, correlation_btw_freq, gamma_val, final_state)

    isr = IsrFormulation(sos, extra_terms, print_extra_term_dict=True)
    print(
        f"The SOS expression was transformed into the following ADC/ISR formulation:\n{isr}\nThus, "
        f"{isr.number_of_extra_terms} non-vanishing terms were identified that must be additionally "
        "considered due to the definition of the ADC matrices.\n")
    print(
        "Building tree to determine suitable response vectors ..."
    )
    rvecs_dict_list = build_tree(isr.mod_expr)

    # prepare the projection of the states excluded from the summation
    to_be_projected_out = []
    for exstate in sos.excluded_states:
        if exstate == O:
            continue  # the ADC quantities do not include the ground state anyway
        elif isinstance(exstate, int):
            to_be_projected_out.append(exstate)
        else:
            assert final_state is not None
            assert exstate == final_state[0]
            to_be_projected_out.append(final_state[1])
    if to_be_projected_out:
        print(f"The following states are projected out from the ADC matrices: {to_be_projected_out}")
        if len(to_be_projected_out) != 1:
            raise NotImplementedError(
                "It is not yet possible to project out more than one state."
            )
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

    rvecs_dict_tot = {}
    response_dict = {}
    number_of_unique_rvecs = 0
    print("Solving response equations ...")
    for tup in rvecs_dict_list:
        root_expr, rvecs_dict = tup
        # check if response equations become equal after inserting values for omegas and gamma
        rvecs_dict_mod = {}
        for key, value in rvecs_dict.items():
            om = float(key[2].subs(all_omegas))
            gam = float(im(key[3].subs(gamma, gamma_val)))
            if gam == 0 and gamma_val != 0:
                raise ValueError(
                        "Although the entered SOS expression is real, a value for gamma was specified."
                )
            new_key = (*key[:2], om, gam, *key[4:])
            if new_key not in rvecs_dict_mod.keys():
                rvecs_dict_mod[new_key] = [value]
            else:
                rvecs_dict_mod[new_key].append(value)
        number_of_unique_rvecs += len(rvecs_dict_mod)
        # solve response equations
        for key, value in rvecs_dict_mod.items():
            op_type = key[1]
            if key[0] == "MTM":
                rhss = np.array(adcc_prop_dict[op_type].mtms)
                op_dim = adcc_prop_dict[op_type].op_dim
                response_shape = (3,)*op_dim
                iterables = [list(range(shape)) for shape in response_shape]
                components = list(product(*iterables))
                response = np.empty(response_shape, dtype=object)
                if key[3] == 0.0:
                    for c in components:
                        response[c] = solve_response(
                                matrix, rhss[c], -key[2], gamma=0.0, projection=projection, **solver_args
                        )
                else:
                    for c in components:
                        response[c] = solve_response(
                                matrix, RV(rhss[c]), -key[2], gamma=-key[3], projection=projection, **solver_args
                        )
                for vv in value:
                    response_dict[vv] = response
            elif key[0] == "S2S_MTM":
                if projection is not None:
                    raise NotImplementedError("It is not yet possible to project out states from the B matrix.")
                dips = np.array(adcc_prop_dict[op_type].dips)
                op_dim = adcc_prop_dict[op_type].op_dim
                if key[4] == "ResponseVector":
                    no = key[5]
                    rvecs = response_dict[no]
                    if key[3] == 0.0:
                        product_vecs_shape = (3,)*op_dim + rvecs.shape
                        iterables = [list(range(shape)) for shape in product_vecs_shape]
                        components = list(product(*iterables))
                        response = np.empty(product_vecs_shape, dtype=object)
                        for c in components:
                            rhs = bmatrix_vector_product(property_method, mp, dips[c[:op_dim]], rvecs[c[op_dim:]])
                            response[c] = solve_response(matrix, rhs, -key[2], gamma=-key[3], **solver_args)
                    else:
                        # complex bmatrix vector product is implemented (but not tested),
                        # but solving response equations with complex right-hand sides is not yet possible
                        raise NotImplementedError("The case of complex response vectors (leading to complex"
                                                  "right-hand sides when solving the response equations)"
                                                  "has not yet been implemented.")
                    for vv in value:
                        response_dict[vv] = response
                elif key[4] == final_state[0]:
                    product_vecs_shape = (3,)*op_dim
                    iterables = [list(range(shape)) for shape in product_vecs_shape]
                    components = list(product(*iterables))
                    response = np.empty(product_vecs_shape, dtype=object)
                    if key[3] == 0.0:
                        for c in components:
                            product_vec = bmatrix_vector_product(
                                    property_method, mp, dips[c], state.excitation_vector[final_state[1]]
                            )
                            response[c] = solve_response(matrix, product_vec, -key[2], gamma=0.0, **solver_args)
                    else:
                        for c in components:
                            product_vec = bmatrix_vector_product(
                                    property_method, mp, dips[c], state.excitation_vector[final_state[1]]
                            )
                            response[c] = solve_response(
                                    matrix, RV(product_vec), -key[2], gamma=-key[3], **solver_args
                            )
                    for vv in value:
                        response_dict[vv] = response
                else:
                    raise ValueError("Unkown response equation.")
            else:
                raise ValueError("Unkown response equation.")
        rvecs_dict_tot.update(dict((value, key) for key, value in rvecs_dict.items()))

    print(f"In total, {len(rvecs_dict_tot)} response vectors (with 3 components each) were defined:")
    for key, value in rvecs_dict_tot.items():
        print(key, ": ", value)
    if len(rvecs_dict_tot) == number_of_unique_rvecs:
        print(f"Thus, {3*number_of_unique_rvecs} response equations had to be solved.\n")
    elif len(rvecs_dict_tot) > number_of_unique_rvecs:
        print(
            "However, inserting the specified frequency values caused response vectors to become equal, "
            f"so that in the end only 3x{number_of_unique_rvecs} response equations had to be solved.\n"
        )
    else:
        raise ValueError()

    if rvecs_dict_list:
        root_expr = rvecs_dict_list[-1][0]
    else:
        root_expr = isr.mod_expr

    dtype = float
    if gamma_val != 0.0:
        dtype = complex
    res_tens = np.zeros((3,)*sos.order, dtype=dtype)

    if isinstance(root_expr, Add):
        term_list = [arg for arg in root_expr.args]
    else:
        term_list = [root_expr]

    if symmetric:
        components = list(combinations_with_replacement([0, 1, 2], sos.order))  # if tensor is symmetric
    else:
        components = list(product([0, 1, 2], repeat=sos.order))
    for c in components:
        comp_map = {
                ABC[ic]: cc for ic, cc in enumerate(c)
        }
        # subs_dict = {o[0]: o[1] for o in all_omegas}
        # subs_dict[gamma] = gamma_val
        for term in term_list:
            subs_dict = {o[0]: o[1] for o in all_omegas}
            subs_dict[gamma] = gamma_val
            for i, a in enumerate(term.args):
                oper_a = a
                if isinstance(a, adjoint):
                    oper_a = a.args[0]
                if isinstance(oper_a, ResponseVector) and oper_a == a:  # vec * X
                    comps_right_v = tuple([comp_map[char] for char in list(oper_a.comp)])
                    right_v = response_dict[oper_a.no][comps_right_v]

                    lhs = term.args[i-1]
                    if isinstance(lhs, S2S_MTM):  # vec * B * X --> transition polarizability
                        dips = np.array(adcc_prop_dict[lhs.op_type].dips)
                        lhs2 = term.args[i-2]
                        key = lhs2*lhs*a
                        if isinstance(lhs2, adjoint) and isinstance(lhs2.args[0], ResponseVector):  # Dagger(X) * B * X
                            comps_left_v = tuple([comp_map[char] for char in list(lhs2.args[0].comp)])
                            if sign_change(lhs2.args[0].no, rvecs_dict_tot):
                                left_v = -1.0 * response_dict[lhs2.args[0].no][comps_left_v]
                            else:
                                left_v = response_dict[lhs2.args[0].no][comps_left_v]
                        elif isinstance(lhs2, Bra):  # <f| * B * X
                            assert lhs2.label[0] == final_state[0]
                            left_v = state.excitation_vector[final_state[1]]
                        else:
                            raise ValueError("Expression cannot be evaluated.")
                        comps_dip = tuple([comp_map[char] for char in list(lhs.comp)])
                        if isinstance(left_v, AmplitudeVector) and isinstance(right_v, AmplitudeVector):
                            subs_dict[key] = transition_polarizability(
                                    property_method, mp, right_v, dips[comps_dip], left_v  # TODO: correct order?
                            )
                        else:
                            if isinstance(left_v, AmplitudeVector):
                                left_v = RV(left_v)
                            subs_dict[key] = transition_polarizability_complex(
                                    property_method, mp, right_v, dips[comps_dip], left_v  # TODO: correct order?
                            )
                    elif isinstance(lhs, adjoint) and isinstance(lhs.args[0], MTM):  # Dagger(F) * X
                        comps_left_v = tuple([comp_map[char] for char in list(lhs.args[0].comp)])
                        if lhs.args[0].symmetry == 1:  # Hermitian operators
                            left_v = np.array(adcc_prop_dict[lhs.args[0].op_type].mtms)
                        elif lhs.args[0].symmetry == 2:  # anti-Hermitian operators
                            left_v = -1.0 * np.array(adcc_prop_dict[lhs.args[0].op_type].mtms)
                        else:
                            raise NotImplementedError("Only Hermitian and anti-Hermitian operators are implemented.")
                        subs_dict[lhs*a] = scalar_product(
                                left_v[comps_left_v], right_v
                        )
                    elif isinstance(lhs, adjoint) and isinstance(lhs.args[0], ResponseVector):  # Dagger(X) * X
                        comps_left_v = tuple([comp_map[char] for char in list(lhs.args[0].comp)])
                        if sign_change(lhs.args[0].no, rvecs_dict_tot):
                            left_v = -1.0 * response_dict[lhs.args[0].no][comps_left_v]
                        else:
                            left_v = response_dict[lhs.args[0].no][comps_left_v]
                        subs_dict[lhs*a] = scalar_product(
                                left_v, right_v
                        )
                    else:
                        raise ValueError("Expression cannot be evaluated.")
                elif isinstance(oper_a, ResponseVector) and oper_a != a:  # Dagger(X) * vec
                    rhs = term.args[i+1]
                    comps_left_v = tuple([comp_map[char] for char in list(oper_a.comp)])
                    if sign_change(oper_a.no, rvecs_dict_tot):
                        left_v = -1.0 * response_dict[oper_a.no][comps_left_v]
                    else:
                        left_v = response_dict[oper_a.no][comps_left_v]

                    if isinstance(rhs, S2S_MTM):  # Dagger(X) * B * vec --> transition polarizability
                        dips = np.array(adcc_prop_dict[rhs.op_type].dips)
                        rhs2 = term.args[i+2]
                        key = a*rhs*rhs2
                        if isinstance(rhs2, ResponseVector):  # Dagger(X) * B * X (taken care of above)
                            continue
                        elif isinstance(rhs2, Ket):  # Dagger(X) * B * |f>
                            assert rhs2.label[0] == final_state[0]
                            right_v = state.excitation_vector[final_state[1]]
                        else:
                            raise ValueError("Expression cannot be evaluated.")
                        comps_dip = tuple([comp_map[char] for char in list(rhs.comp)])
                        if isinstance(left_v, AmplitudeVector) and isinstance(right_v, AmplitudeVector):
                            subs_dict[key] = transition_polarizability(
                                    property_method, mp, right_v, dips[comps_dip], left_v
                            )
                        else:
                            right_v = RV(right_v)
                            subs_dict[key] = transition_polarizability_complex(
                                    property_method, mp, right_v, dips[comps_dip], left_v
                            )
                    elif isinstance(rhs, MTM):  # Dagger(X) * F
                        comps_right_v = tuple([comp_map[char] for char in list(rhs.comp)])
                        right_v = np.array(adcc_prop_dict[rhs.op_type].mtms)
                        subs_dict[a*rhs] = scalar_product(
                                left_v, right_v[comps_right_v]
                        )
                    elif isinstance(rhs, ResponseVector):  # Dagger(X) * X (taken care of above)
                        continue
                    else:
                        raise ValueError("Expression cannot be evaluated.")

                elif isinstance(a, DipoleMoment):
                    comps_dipmom = tuple([comp_map[char] for char in list(a.comp)])
                    if a.from_state == O and a.to_state == O:
                        gs_dip_moment = adcc_prop_dict[a.op_type].gs_dip_moment
                        subs_dict[a] = gs_dip_moment[comps_dipmom]
                    elif a.from_state == O and a.to_state == final_state[0]:
                        tdms = adcc_prop_dict[a.op_type].transition_dipole_moment
                        subs_dict[a] = tdms[final_state[1]][comps_dipmom]
                    else:
                        raise ValueError("Unknown dipole moment.")
                elif isinstance(a, LeviCivita):
                    subs_dict[a] = lc_tensor[c]
            res = term.subs(subs_dict)
            if res == zoo:
                raise ZeroDivisionError()
            res_tens[c] += res
        # print(root_expr, subs_dict)
        # res = root_expr.subs(subs_dict)
        # print(res)
        # if res == zoo:
        #     raise ZeroDivisionError()
        # res_tens[c] = res
        if symmetric:
            perms = list(permutations(c))  # if tensor is symmetric
            for pe in perms:
                res_tens[pe] = res_tens[c]
    print("========== The requested tensor was formed. ==========")
    return res_tens


def evaluate_property_sos(
        state, sos_expr, summation_indices, omegas=None, gamma_val=0.0,
        final_state=None, perm_pairs=None, extra_terms=True, symmetric=False, excluded_states=None):
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

    omegas: list of tuples, optional
        List of (symbol, value) pairs for the frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'>
        or <class 'sympy.core.symbol.Symbol'> or float),
        e.g., [(w_o, w_1+w_2), (w_1, 0.5), (w_2, 0.5)].

    gamma_val: float, optional

    final_state: tuple, optional
        (<class 'sympy.core.symbol.Symbol'>, int), e.g., (f, 0).

    perm_pairs: list of tuples, optional
        List of (op, freq) pairs whose permutation yields the full SOS expression;
        (op, freq): (<class 'responsetree.response_operators.DipoleOperator'>, <class 'sympy.core.symbol.Symbol'>),
        e.g., [(op_a, -w_o), (op_b, w_1), (op_c, w_2)].

    extra_terms: bool, optional
        Compute the additional terms that arise when converting the SOS expression to its ADC/ISR formulation;
        by default 'True'.

    symmetric: bool, optional
        Resulting tensor is symmetric;
        by default 'False'.

    excluded_states: list of <class 'sympy.core.symbol.Symbol'> or int, optional
        List of states that are excluded from the summation.
        It is important to note that the ground state is represented by the SymPy symbol O, while the integer 0
        represents the first excited state.

    Returns
    ----------
    <class 'numpy.ndarray'>
        Resulting tensor.
    """
    if omegas is None:
        omegas = []
    elif type(omegas) == tuple:
        omegas = [omegas]
    else:
        assert type(omegas) == list
    assert type(extra_terms) == bool
    assert type(symmetric) == bool

    # create SumOverStates object from input
    correlation_btw_freq = [tup for tup in omegas if type(tup[1]) == Symbol or type(tup[1]) == Add]
    sos = SumOverStates(
            sos_expr, summation_indices, correlation_btw_freq=correlation_btw_freq,
            perm_pairs=perm_pairs, excluded_states=excluded_states
    )
    print(
        f"\nThe following SOS expression was entered/generated. It consists of {sos.number_of_terms} term(s):\n{sos}\n"
    )
    # store adcc properties for the required operators in a dict
    adcc_prop_dict = {}
    for op_type in sos.operator_types:
        adcc_prop_dict[op_type] = AdccProperties(state, op_type)

    all_omegas = omegas.copy()
    if final_state:
        assert type(final_state) == tuple and len(final_state) == 2
        all_omegas.append(
                (TransitionFrequency(final_state[0], real=True),
                 state.excitation_energy_uncorrected[final_state[1]])
        )
        for ies, exstate in enumerate(sos.excluded_states):
            if isinstance(exstate, int) and exstate == final_state[1]:
                sos.excluded_states[ies] = final_state[0]
    else:
        assert final_state is None

    _check_omegas_and_final_state(sos.expr, omegas, sos.correlation_btw_freq, gamma_val, final_state)

    # all terms are stored as dictionaries in a list
    if isinstance(sos.expr, Add):
        term_list = [
                {"expr": term, "summation_indices": sos.summation_indices,
                 "transition_frequencies": sos.transition_frequencies}
                for term in sos.expr.args
        ]
    else:
        term_list = [
                {"expr": sos.expr, "summation_indices": sos.summation_indices,
                 "transition_frequencies": sos.transition_frequencies}
        ]
    if extra_terms:
        print("Determining extra terms ...")
        ets = compute_extra_terms(
                sos.expr, sos.summation_indices, excluded_states=sos.excluded_states,
                correlation_btw_freq=sos.correlation_btw_freq, print_extra_term_dict=True
        )
        if isinstance(ets, Add):
            et_list = list(ets.args)
        elif isinstance(ets, Mul):
            et_list = [ets]
        else:
            et_list = []
        print(
            f"{len(et_list)} non-vanishing terms were identified that must be additionally considered "
            "due to the definition of the adcc properties.\n"
        )
        for et in et_list:
            # the extra terms contain less indices of summation
            sum_ind = find_remaining_indices(et, sos.summation_indices)
            trans_freq = [TransitionFrequency(index, real=True) for index in sum_ind]
            term_list.append(
                    {"expr": et, "summation_indices": sum_ind, "transition_frequencies": trans_freq}
            )

    if final_state:
        for ies, exstate in enumerate(sos.excluded_states):
            if exstate == final_state[0]:
                sos.excluded_states[ies] = final_state[1]

    dtype = float
    if gamma_val != 0.0:
        dtype = complex
    res_tens = np.zeros((3,)*sos.order, dtype=dtype)

    if symmetric:
        components = list(combinations_with_replacement([0, 1, 2], sos.order))  # if tensor is symmetric
    else:
        components = list(product([0, 1, 2], repeat=sos.order))

    print(f"Summing over {len(state.excitation_energy_uncorrected)} excited states ...")
    for term_dict in tqdm(term_list):
        mod_expr = replace_bra_op_ket(
                term_dict["expr"].subs(sos.correlation_btw_freq)
        )
        sum_ind = term_dict["summation_indices"]

        # values that the indices of summation can take on
        indices = list(
                product(range(len(state.excitation_energy_uncorrected)), repeat=len(sum_ind))
        )
        dip_mom_list = [a for a in mod_expr.args if isinstance(a, DipoleMoment)]
        lc_contained = False
        for a in mod_expr.args:
            if isinstance(a, LeviCivita):
                lc_contained = True
        for i in indices:
            state_map = {
                    sum_ind[ii]: ind for ii, ind in enumerate(i)
                }

            # skip the rest of the loop for this iteration if it corresponds to one of the excluded states
            if set(sos.excluded_states).intersection(set(state_map.values())):
                continue

            if final_state:
                state_map[final_state[0]] = final_state[1]
            for c in components:
                comp_map = {
                        ABC[ic]: cc for ic, cc in enumerate(c)
                }
                subs_dict = {o[0]: o[1] for o in all_omegas}
                subs_dict[gamma] = gamma_val

                for si, tf in zip(sum_ind, term_dict["transition_frequencies"]):
                    subs_dict[tf] = state.excitation_energy_uncorrected[state_map[si]]

                for a in dip_mom_list:
                    comps_dipmom = tuple([comp_map[char] for char in list(a.comp)])
                    if a.from_state == O and a.to_state == O:
                        gs_dip_moment = adcc_prop_dict[a.op_type].gs_dip_moment
                        subs_dict[a] = gs_dip_moment[comps_dipmom]
                    elif a.from_state == O:
                        index = state_map[a.to_state]
                        tdms = adcc_prop_dict[a.op_type].transition_dipole_moment
                        subs_dict[a] = tdms[index][comps_dipmom]
                    elif a.to_state == O:
                        index = state_map[a.from_state]
                        tdms = adcc_prop_dict[a.op_type].transition_dipole_moment
                        if a.symmetry == 1:  # Hermitian operators
                            subs_dict[a] = tdms[index][comps_dipmom]
                        elif a.symmetry == 2:  # anti-Hermitian operators
                            subs_dict[a] = -1.0 * tdms[index][comps_dipmom]  # TODO: correct sign?
                        else:
                            raise NotImplementedError("Only Hermitian and anti-Hermitian operators are implemented.")
                    else:
                        index1 = state_map[a.from_state]
                        index2 = state_map[a.to_state]
                        if a.from_state in sum_ind and a.to_state in sum_ind:  # e.g., <n|op|m>
                            s2s_tdms = adcc_prop_dict[a.op_type].state_to_state_transition_moment
                            subs_dict[a] = s2s_tdms[index1, index2][comps_dipmom]
                        elif a.from_state in sum_ind:  # e.g., <f|op|n>
                            s2s_tdms_f = adcc_prop_dict[a.op_type].s2s_tm(final_state=index2)
                            subs_dict[a] = s2s_tdms_f[index1][comps_dipmom]
                        elif a.to_state in sum_ind:  # e.g., <n|op|f>
                            s2s_tdms_f = adcc_prop_dict[a.op_type].s2s_tm(initial_state=index1)
                            subs_dict[a] = s2s_tdms_f[index2][comps_dipmom]
                        else:
                            raise ValueError()
                if lc_contained:
                    subs_dict[LeviCivita()] = lc_tensor[c]
                res = mod_expr.xreplace(subs_dict)
                if res == zoo:
                    raise ZeroDivisionError()
                res_tens[c] += res
                if symmetric:
                    perms = list(permutations(c))  # if tensor is symmetric
                    for pe in perms:
                        res_tens[pe] = res_tens[c]
    print("========== The requested tensor was formed. ==========")
    return res_tens


def evaluate_property_sos_fast(
        state, sos_expr, summation_indices, omegas=None, gamma_val=0.0,
        final_state=None, perm_pairs=None, extra_terms=True, excluded_states=None):
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

    omegas: list of tuples, optional
        List of (symbol, value) pairs for the frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'>
        or <class 'sympy.core.symbol.Symbol'> or float),
        e.g., [(w_o, w_1+w_2), (w_1, 0.5), (w_2, 0.5)].

    gamma_val: float, optional

    final_state: tuple, optional
        (<class 'sympy.core.symbol.Symbol'>, int), e.g., (f, 0).

    perm_pairs: list of tuples, optional
        List of (op, freq) pairs whose permutation yields the full SOS expression;
        (op, freq): (<class 'responsetree.response_operators.DipoleOperator'>, <class 'sympy.core.symbol.Symbol'>),
        e.g., [(op_a, -w_o), (op_b, w_1), (op_c, w_2)].

    extra_terms: bool, optional
        Compute the additional terms that arise when converting the SOS expression to its ADC/ISR formulation;
        by default 'True'.

    excluded_states: list of <class 'sympy.core.symbol.Symbol'> or int, optional
        List of states that are excluded from the summation.
        It is important to note that the ground state is represented by the SymPy symbol O, while the integer 0
        represents the first excited state.

    Returns
    ----------
    <class 'numpy.ndarray'>
        Resulting tensor.
    """
    if omegas is None:
        omegas = []
    elif type(omegas) == tuple:
        omegas = [omegas]
    else:
        assert type(omegas) == list
    assert type(extra_terms) == bool

    # create SumOverStates object from input
    correlation_btw_freq = [tup for tup in omegas if type(tup[1]) == Symbol or type(tup[1]) == Add]
    sos = SumOverStates(
            sos_expr, summation_indices, correlation_btw_freq=correlation_btw_freq,
            perm_pairs=perm_pairs, excluded_states=excluded_states
    )
    print(
        f"\nThe following SOS expression was entered/generated. It consists of {sos.number_of_terms} term(s):\n{sos}\n"
    )
    # store adcc properties for the required operators in a dict
    adcc_prop_dict = {}
    for op_type in sos.operator_types:
        adcc_prop_dict[op_type] = AdccProperties(state, op_type)

    subs_dict = {om_tup[0]: om_tup[1] for om_tup in omegas}
    if final_state:
        assert type(final_state) == tuple and len(final_state) == 2
        subs_dict[TransitionFrequency(final_state[0], real=True)] = (
            state.excitation_energy_uncorrected[final_state[1]]
        )
        for ies, exstate in enumerate(sos.excluded_states):
            if isinstance(exstate, int) and exstate == final_state[1]:
                sos.excluded_states[ies] = final_state[0]
    else:
        assert final_state is None
    subs_dict[gamma] = gamma_val

    _check_omegas_and_final_state(sos.expr, omegas, correlation_btw_freq, gamma_val, final_state)

    if extra_terms:
        print("Determining extra terms ...")
        computed_terms = compute_extra_terms(
                sos.expr, sos.summation_indices, excluded_states=sos.excluded_states,
                correlation_btw_freq=sos.correlation_btw_freq, print_extra_term_dict=True
        )
        if computed_terms == 0:
            number_of_extra_terms = 0
        elif isinstance(computed_terms, Add):
            number_of_extra_terms = len(computed_terms.args)
        else:
            number_of_extra_terms = 1
        print(
            f"{number_of_extra_terms} non-vanishing terms were identified that must be additionally "
            "considered due to the definition of the adcc properties.\n"
        )
        sos_with_et = sos.expr + computed_terms
        sos_expr_mod = sos_with_et.subs(correlation_btw_freq)
    else:
        sos_expr_mod = sos.expr.subs(correlation_btw_freq)

    dtype = float
    if gamma_val != 0.0:
        dtype = complex
    res_tens = np.zeros((3,)*sos.order, dtype=dtype)

    if isinstance(sos_expr_mod, Add):
        term_list = [replace_bra_op_ket(arg) for arg in sos_expr_mod.args]
    else:
        term_list = [replace_bra_op_ket(sos_expr_mod)]
    print(
        f"Summing over {len(state.excitation_energy_uncorrected)} excited states using the Einstein "
        "summation convention ..."
    )
    for it, term in enumerate(term_list):
        einsum_list = []
        factor = 1
        divergences = []
        for a in term.args:
            if isinstance(a, DipoleMoment):
                if a.from_state == O and a.to_state == O:  # <0|op|0>
                    gs_dip_moment = adcc_prop_dict[a.op_type].gs_dip_moment
                    einsum_list.append(("", a.comp, gs_dip_moment))
                elif a.from_state == O:
                    tdms = adcc_prop_dict[a.op_type].transition_dipole_moment  # TODO: correct sign?
                    if a.to_state in sos.summation_indices:  # e.g., <n|op|0>
                        einsum_list.append((str(a.to_state), a.comp, tdms))
                    else:  # e.g., <f|op|0>
                        einsum_list.append(("", a.comp, tdms[final_state[1]]))
                elif a.to_state == O:
                    if a.symmetry == 1:  # Hermitian operators
                        tdms = adcc_prop_dict[a.op_type].transition_dipole_moment
                    elif a.symmetry == 2:  # anti-Hermitian operators
                        tdms = -1.0 * adcc_prop_dict[a.op_type].transition_dipole_moment  # TODO: correct sign?
                    else:
                        raise NotImplementedError("Only Hermitian and anti-Hermitian operators are implemented.")
                    if a.from_state in sos.summation_indices:  # e.g., <0|op|n>
                        einsum_list.append((str(a.from_state), a.comp, tdms))
                    else:  # e.g., <0|op|f>
                        einsum_list.append(("", a.comp, tdms[final_state[1]]))
                else:
                    if a.from_state in sos.summation_indices and a.to_state in sos.summation_indices:  # e.g., <n|op|m>
                        s2s_tdms = adcc_prop_dict[a.op_type].state_to_state_transition_moment
                        einsum_list.append((str(a.from_state)+str(a.to_state), a.comp, s2s_tdms))
                    elif a.from_state in sos.summation_indices and a.to_state == final_state[0]:  # e.g., <f|op|n>
                        s2s_tdms_f = adcc_prop_dict[a.op_type].s2s_tm(final_state=final_state[1])
                        einsum_list.append((str(a.from_state), a.comp, s2s_tdms_f))
                    elif a.to_state in sos.summation_indices and a.from_state == final_state[0]:  # e.g., <n|op|f>
                        s2s_tdms_f = adcc_prop_dict[a.op_type].s2s_tm(initial_state=final_state[1])
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
                        shift += 1j*float(aa.args[0])
                    else:
                        raise ValueError()
                if index is None:
                    if shift:
                        einsum_list.append(("", "", 1/(shift)))
                    else:
                        raise ZeroDivisionError()
                else:
                    array = 1/(state.excitation_energy_uncorrected + shift)
                    if np.inf in array:
                        index_with_inf = np.where(array == np.inf)
                        assert len(index_with_inf) == 1
                        assert len(index_with_inf[0]) == 1
                        divergences.append((index, index_with_inf[0][0]))
                    einsum_list.append((str(index), "", array))

            elif isinstance(a, LeviCivita):
                einsum_list.append(("", "ABC", lc_tensor))

            elif isinstance(a, Integer) or isinstance(a, Float):
                factor *= float(a)

            else:
                raise TypeError(f"The following type was not recognized: {type(a)}.")

        if len(divergences) != 0:
            print("The following divergences have been found (explaining the RuntimeWarning): ", divergences)
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
                        assert final_state is not None
                        assert exstate == final_state[0]
                        index_to_delete = final_state[1]
                    for axis in range(len(state_str)):
                        array = np.delete(array, index_to_delete, axis=axis)
                        removed_divergences.append((Symbol(state_str[axis], real=True), index_to_delete))
            array_list.append(array)
        removed_divergences = list(set(removed_divergences))
        divergences_copied = divergences.copy()
        for rd in removed_divergences:
            if rd in divergences:
                divergences_copied.remove(rd)
        if len(divergences) != 0:
            if len(divergences_copied) != 0:
                raise ZeroDivisionError("Not all divergences that occured could be eliminated."
                                        f"The following divergences remain: {divergences}.")
            else:
                print("However, all of these divergences have been successfully removed.")
        einsum_left_mod = einsum_left[:-1]
        einsum_right_list = list(set(einsum_right))
        einsum_right_list.sort()
        einsum_right_mod = ''.join(einsum_right_list)
        einsum_string = einsum_left_mod + " -> " + einsum_right_mod
        print(f"Created string of subscript labels that is used by np.einsum for term {it+1}:\n", einsum_string)
        res_tens += (factor * np.einsum(einsum_string, *array_list))

    print("========== The requested tensor was formed. ==========")
    return res_tens


if __name__ == "__main__":
    from pyscf import gto, scf
    import adcc
    from responsefun.symbols_and_labels import (
            op_a, op_b, op_c, op_d,
            opm_b,
            n, m, k, p, f,
            w_n, w_m, w_k, w_p, w_f,
            w, w_1, w_2, w_3, w_o
    )
    from responsefun.testdata import cache
    from responsefun.test_property import SOS_expressions

    mol = gto.M(
        atom="""
        O 0 0 0
        H 0 0 1.795239827225189
        H 1.693194615993441 0 -0.599043184453037
        """,
        unit="Bohr",
        basis="sto-3g",
    )

    scfres = scf.RHF(mol)
    scfres.kernel()

    refstate = adcc.ReferenceState(scfres)
    matrix = adcc.AdcMatrix("adc2", refstate)
    state = adcc.adc2(scfres, n_singlets=65)
    mock_state = cache.data_fulldiag["h2o_sto3g_adc2"]

    alpha_term = SOS_expressions['alpha_complex'][0]
    omega_alpha = [(w, 0.5)]
    gamma_val = 0.01
    alpha_tens = evaluate_property_isr(state, alpha_term, [n], omega_alpha, gamma_val=gamma_val)
    # print(alpha_tens)
    # alpha_tens_ref = complex_polarizability(refstate, "adc2", 0.5, gamma_val)
    # print(alpha_tens_ref)

    beta_term = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, opm_b, k) * TransitionMoment(k, op_c, O)
            / ((w_n - w_o) * (w_k - w_2))
    )
    # beta_mag_isr = evaluate_property_isr(
    #     state, beta_term, [n,k], [(w_o, w_1+w_2), (w_1, 0.5), (w_2, 0.5)],
    #     perm_pairs=[(op_a, -w_o), (opm_b, w_1), (op_c, w_2)]
    # )
    # beta_mag_sos = evaluate_property_sos_fast(
    #     state, beta_term, [n,k], [(w_o, w_1+w_2), (w_1, 0.5), (w_2, 0.5)],
    #     perm_pairs=[(op_a, -w_o), (opm_b, w_1), (op_c, w_2)]
    # )
    # print(beta_mag_isr)
    # print(beta_mag_sos)
    # np.testing.assert_allclose(beta_mag_isr, beta_mag_sos, atol=1e-8)

    gamma_term = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m)
            * TransitionMoment(m, op_c, p) * TransitionMoment(p, op_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    )
    gamma_omegas = [(w_1, 0.5), (w_2, 0.55), (w_3, 0.6), (w_o, w_1+w_2+w_3)]
    # gamma_tens1 = evaluate_property_isr(
    #         state, gamma_term, [m, n, p], gamma_omegas, extra_terms=False
    # )
    # print(gamma_tens1)
    # gamma_tens1_sos = (
    #         evaluate_property_sos_fast(mock_state, gamma_term, [m, n, p], gamma_omegas, extra_terms=False)
    # )
    # print(gamma_tens1_sos)
    # np.testing.assert_allclose(gamma_tens1, gamma_tens1_sos, atol=1e-6)

    threepa_term = (
            TransitionMoment(O, op_a, m) * TransitionMoment(m, op_b, n) * TransitionMoment(n, op_c, f)
            / ((w_n - w_1 - w_2) * (w_m - w_1))
    )
    # threepa_perm_pairs = [(op_a, w_1), (op_b, w_2), (op_c, w_3)]
    # threepa_omegas = [
    #         (w_1, state.excitation_energy[0]/3),
    #         (w_2, state.excitation_energy[0]/3),
    #         (w_3, state.excitation_energy[0]/3),
    #         (w_1, w_f-w_2-w_3)
    # ]
    # threepa_tens = (
    #         evaluate_property_isr(state, threepa_term, [m, n], threepa_omegas,
    #         perm_pairs=threepa_perm_pairs, final_state=(f, 0))
    # )
    # print(threepa_tens)
    # threepa_term = (
    #         TransitionMoment(O, op_a, m) * TransitionMoment(m, op_b, n) * TransitionMoment(n, op_c, f)
    #         / ((w_n - 2*(w_f/3)) * (w_m - (w_f/3)))
    # )
    # threepa_perm_pairs = [(op_a, w), (op_b, w), (op_c, w)]
    # threepa_omegas = [
    #         #(w, state.excitation_energy[0]/3),
    #         #(w, w_f/3)
    # ]
    # threepa_tens = (
    #         evaluate_property_isr(state, threepa_term, [m, n], threepa_omegas,
    #         perm_pairs=threepa_perm_pairs, final_state=(f, 0))
    # )
    # print(threepa_tens)

    # threepa_tens_sos = (
    #         evaluate_property_sos_fast(state, threepa_term, [m, n], threepa_omegas,
    #         perm_pairs=threepa_perm_pairs, final_state=(f, 0))
    # )
    # print(threepa_tens_sos)
    # np.testing.assert_allclose(threepa_tens, threepa_tens_sos, atol=1e-6)

    # TODO: make it work for esp also in the static case --> projecting the fth eigenstate out of the matrix
    omega_alpha = [(w, 0)]
    esp_terms = (
        TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - w_f - w - 1j*gamma)
        + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - w_f + w + 1j*gamma)
    )
    # esp_tens = evaluate_property_isr(
    #         state, esp_terms, [n], omega_alpha, 0.0/Hartree, final_state=(f, 0), excluded_states=f
    # )
    # print(esp_tens)
    # esp_tens_sos = evaluate_property_sos_fast(
    #         mock_state, esp_terms, [n], omega_alpha, 0.0/Hartree, final_state=(f, 0), excluded_states=f
    # )
    # print(esp_tens_sos)
    # np.testing.assert_allclose(esp_tens, esp_tens_sos, atol=1e-7)

    epsilon = LeviCivita()
    mcd_term1 = (
            -1.0 * epsilon
            * TransitionMoment(O, opm_b, k) * TransitionMoment(k, op_c, f) * TransitionMoment(f, op_a, O)
            / w_k
    )
    # mcd_tens1 = evaluate_property_isr(
    #         state, mcd_term1, [k], final_state=(f, 0), extra_terms=False, excluded_states=[O]
    # )
    # print(mcd_tens1)
    # mcd_tens1_sos = evaluate_property_sos_fast(
    #         mock_state, mcd_term1, [k], final_state=(f, 0), extra_terms=False, excluded_states=[O]
    # )
    # print(mcd_tens1_sos)
    # np.testing.assert_allclose(mcd_tens1, mcd_tens1_sos, atol=1e-12)
    # mcd_tens1_sos2 = evaluate_property_sos(
    #         state, mcd_term1, [k], final_state=(f, 0), extra_terms=False, excluded_cases=[(k, O)]
    # )
    # print(mcd_tens1_sos2)
    # np.testing.assert_allclose(mcd_tens1, mcd_tens1_sos, atol=1e-7)
    # np.testing.assert_allclose(mcd_tens1_sos, mcd_tens1_sos2, atol=1e-7)
    mcd_term2 = (
            -1.0 * epsilon
            * TransitionMoment(O, op_c, k) * TransitionMoment(k, opm_b, f) * TransitionMoment(f, op_a, O)
            / (w_k - w_f)
    )
    # mcd_tens2 = evaluate_property_isr(
    #         state, mcd_term2, [k], final_state=(f, 0), extra_terms=False, excluded_states=[O, f]
    # )
    # print(mcd_tens2)
    # mcd_tens2_sos = evaluate_property_sos_fast(
    #         mock_state, mcd_term2, [k], final_state=(f, 0), extra_terms=False, excluded_states=[O, f]
    # )
    # print(mcd_tens2_sos)
    # mcd_tens2_sos2 = evaluate_property_sos(
    #         mock_state, mcd_term2, [k], final_state=(f, 0), extra_terms=False, excluded_states=[O, 0]
    # )
    # print(mcd_tens2_sos2)
    # np.testing.assert_allclose(mcd_tens2, mcd_tens2_sos, atol=1e-7)
    # mcd_tens = mcd_tens1+mcd_tens2
    # mcd_tens2 = mcd_tens1_sos+mcd_tens2_sos
    # print(mcd_tens)
    # np.testing.assert_allclose(mcd_tens, mcd_tens2, atol=1e-7)

    # excited_state = Excitation(state, 0, "adc2")
    # mcd_ref = mcd_bterm(excited_state)
    # print(mcd_ref)

    gamma_extra_term = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O)
            * TransitionMoment(O, op_c, m) * TransitionMoment(m, op_d, O)
            / ((w_n - w_o) * (w_m - w_3) * (w_m + w_2))
    )
    # gamma_extra_tens = evaluate_property_isr(
    #         state, gamma_extra_term, [n, m], omegas=[(w_1, 0.5), (w_2, 0.6), (w_3, 0.7), (w_o, w_1+w_2+w_3)],
    #         perm_pairs=[(op_a, -w_o), (op_b, w_1), (op_c, w_2), (op_d, w_3)],
    #         extra_terms=False
    # )
    # print(gamma_extra_tens)

    esp_extra_terms = (
        TransitionMoment(f, op_a, O) * TransitionMoment(O, op_b, f) / (- w_f - w - 1j*gamma)
        + TransitionMoment(f, op_b, O) * TransitionMoment(O, op_a, f) / (- w_f + w + 1j*gamma)
    )
    # esp_extra_tens = evaluate_property_isr(
    #         state, esp_extra_terms, [], omegas=[(w, 0.5)], gamma_val=0.01, final_state=(f, 2), extra_terms=False
    # )
    # print(esp_extra_tens)
