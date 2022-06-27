import numpy as np
import string
from scipy.constants import physical_constants

from sympy.physics.quantum.state import Bra, Ket
from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, im, Float, Integer, S, zoo, I
from itertools import permutations, product, combinations_with_replacement

from responsefun.symbols_and_labels import *
from responsefun.response_operators import MTM, S2S_MTM, ResponseVector, DipoleOperator, DipoleMoment, TransitionFrequency, LeviCivita
from responsefun.sum_over_states import TransitionMoment, SumOverStates
from responsefun.isr_conversion import to_isr, compute_extra_terms
from responsefun.build_tree import build_tree
from responsefun.testdata.cache import MockExcitedStates
from responsefun.bmatrix_vector_product import bmatrix_vector_product
from responsefun.magnetic_dipole_moments import modified_magnetic_transition_moments, gs_magnetic_dipole_moment

from adcc import AmplitudeVector
from adcc.workflow import construct_adcmatrix
from adcc.adc_pp import modified_transition_moments
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.OneParticleOperator import product_trace
from respondo.misc import select_property_method
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
                raise ValueError("Two different values were given for the same frequency.")

        sum_freq = [freq for tup in correlation_btw_freq for freq in tup[1].args]
        check_dict = {o[0]: False for o in omegas}
        for o in check_dict:
            for denom in denom_list:
                if o in denom.args or -o in denom.args or o in sum_freq or -o in sum_freq:
                    check_dict[o] = True
                    break
        if False in check_dict.values():
            pass
            #raise ValueError(
            #        "A frequency was specified that is not included in the entered SOS expression.\nomegas: {}".format(check_dict)
            #)

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
        if check_f == False:
            raise ValueError("A final state was mistakenly specified.")


def find_indices(sos_expr, summation_indices):
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
    """Replace Bra(from_state)*op*Ket(to_state) sequence in a SymPy term
    by an instance of <class 'responsetree.response_operators.DipoleMoment'>.
    """
    assert type(expr) == Mul
    subs_dict = {}
    for ia, a in enumerate(expr.args):
        if isinstance(a, DipoleOperator):
            from_state = expr.args[ia-1]
            to_state = expr.args[ia+1]
            key = from_state*a*to_state
            subs_dict[key] = DipoleMoment(a.comp, str(from_state.label[0]), str(to_state.label[0]), a.op_type)
    return expr.subs(subs_dict)


def state_to_state_transition_moments(state, op_type, final_state=None):
    if isinstance(state, MockExcitedStates):
        if op_type == "electric":
            tdms_s2s = state.transition_dipole_moment_s2s
        else:
            tdms_s2s = state.transition_magnetic_moment_s2s
        if final_state is None:
            return tdms_s2s
        else:
            return tdms_s2s[:, final_state]
    else:
        if op_type == "electric":
            dips = state.reference_state.operators.electric_dipole
        else:
            dips = state.reference_state.operators.magnetic_dipole
        if final_state is None:
            s2s_tdms = np.zeros((state.size, state.size, 3))
            excitations = state.excitations
        else:
            assert type(final_state) == int
            s2s_tdms = np.zeros((state.size, 1, 3))
            excitations = [state.excitations[final_state]]
        for ee1 in tqdm(state.excitations):
            i = ee1.index
            for j, ee2 in enumerate(excitations):
                tdm = state2state_transition_dm(
                    state.property_method,
                    state.ground_state,
                    ee1.excitation_vector,
                    ee2.excitation_vector,
                    state.matrix.intermediates,
                )
                s2s_tdms[i, j] = np.array([product_trace(tdm, dip) for dip in dips])
        return np.squeeze(s2s_tdms)


def from_vec_to_vec(from_vec, to_vec):
    """Evaluate the scalar product of a vector of modified transition moments and a response vector."""

    if isinstance(from_vec, AmplitudeVector) and isinstance(to_vec, AmplitudeVector):
        return from_vec @ to_vec
    elif isinstance(from_vec, RV) and isinstance(to_vec, AmplitudeVector):
        real = from_vec.real @ to_vec
        imag = from_vec.imag @ to_vec
        return real + 1j*imag
    elif isinstance(from_vec, AmplitudeVector) and isinstance(to_vec, RV):
        real = from_vec @ to_vec.real
        imag = from_vec @ to_vec.imag
        return real + 1j*imag
    else:
        raise ValueError()


def evaluate_property_isr(
        state, sos_expr, summation_indices, omegas=None, gamma_val=0.0,
        final_state=None, perm_pairs=None, extra_terms=True, symmetric=False, excluded_cases=None, **solver_args
    ):
    """Compute a molecular property with the ADC/ISR approach from its SOS expression.

    Parameters
    ----------
    state: <class 'adcc.ExcitedStates.ExcitedStates'>
        ExcitedStates object returned by an ADC calculation.

    sos_expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
        SymPy expression of the SOS;
        it can be either the full expression or a single term from which the full expression can be generated via permutation.

    summation_indices: list of <class 'sympy.core.symbol.Symbol'>
        List of indices of summation.

    omegas: list of tuples, optional
        List of (symbol, value) pairs for the frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'> or <class 'sympy.core.symbol.Symbol'> or float),
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

    Returns
    ----------
    <class 'numpy.ndarray'>
        Resulting tensor.
    """
    matrix = construct_adcmatrix(state.matrix)
    property_method = select_property_method(matrix)
    mp = matrix.ground_state
    dips_el = state.reference_state.operators.electric_dipole
    mtms_el = modified_transition_moments(property_method, mp, dips_el)

    if omegas is None:
        omegas = []
    elif type(omegas) == tuple:
        omegas = [omegas]
    else:
        assert type(omegas) == list
    assert type(symmetric) == bool
    
    correlation_btw_freq = [tup for tup in omegas if type(tup[1]) == Symbol or type(tup[1]) == Add]
    all_omegas = omegas.copy()
    if final_state:
        assert type(final_state) == tuple and len(final_state) == 2
        all_omegas.append(
                (TransitionFrequency(str(final_state[0]), real=True),
                state.excitation_energy_uncorrected[final_state[1]])
        )
    else:
        assert final_state is None
    sos = SumOverStates(
            sos_expr, summation_indices, correlation_btw_freq=correlation_btw_freq, perm_pairs=perm_pairs, excluded_cases=excluded_cases
    )
    if "magnetic" in set([op.op_type for op in sos.operators]):
        dips_mag = state.reference_state.operators.magnetic_dipole
        mtms_mag = modified_magnetic_transition_moments(property_method, mp, dips_mag)
    
    _check_omegas_and_final_state(sos.expr, omegas, correlation_btw_freq, gamma_val, final_state)
    isr = to_isr(sos, extra_terms)
    mod_isr = isr.subs(correlation_btw_freq)
    rvecs_dict_list = build_tree(mod_isr)

    response_dict = {}
    for tup in rvecs_dict_list:
        root_expr, rvecs_dict = tup
        # check if response equations become equal after inserting values for omegas and gamma
        rvecs_dict_mod = {}
        for k, v in rvecs_dict.items():
            om = float(k[2].subs(all_omegas))
            gam = float(im(k[3].subs(gamma, gamma_val)))
            if gam == 0 and gamma_val != 0:
                raise ValueError(
                        "Although the entered SOS expression is real, a value for gamma was specified."
                )
            new_key = (*k[:2], om, gam, *k[4:])
            if new_key not in rvecs_dict_mod.keys():
                rvecs_dict_mod[new_key] = [v]
            else:
                rvecs_dict_mod[new_key].append(v)
        
        # solve response equations
        for k, v in rvecs_dict_mod.items():
            if k[0] == MTM:
                if k[1] == "electric":
                    rhss = mtms_el
                else:
                    rhss = mtms_mag
                if k[3] == 0.0:
                    response = [solve_response(matrix, rhs, -k[2], gamma=0.0, **solver_args) for rhs in rhss]
                else:
                    response = [solve_response(matrix, RV(rhs), -k[2], gamma=-k[3], **solver_args) for rhs in rhss]
                for vv in v:
                    response_dict[vv] = np.array(response, dtype=object)
            elif k[0] == S2S_MTM:
                if k[1] == "electric":
                    dips = dips_el
                else:
                    dips = dips_mag
                if k[4] == ResponseVector:
                    no = k[5]
                    rvecs = response_dict[no]
                    if k[3] == 0.0:
                        product_vecs = bmatrix_vector_product(property_method, mp, dips, rvecs)
                        iterables = [list(range(shape)) for shape in product_vecs.shape]
                        components = list(product(*iterables))
                        response = np.empty(product_vecs.shape, dtype=object)
                        for c in components:
                            rhs = product_vecs[c]
                            response[c] = solve_response(matrix, rhs, -k[2], gamma=-k[3], **solver_args)
                    else:
                        # complex bmatrix vector product is implemented (but not tested),
                        # but solving response equations with complex right-hand sides is not yet possible
                        raise NotImplementedError("The case of complex response vectors (leading to complex right-hand sides"
                                                  "when solving the response equations) has not yet been implemented.")
                    for vv in v:
                        response_dict[vv] = response
                elif k[4] == final_state[0]:
                    product_vecs = bmatrix_vector_product(property_method, mp, dips, state.excitation_vector[final_state[1]])
                    if k[3] == 0.0:
                        response = [solve_response(matrix, rhs, -k[2], gamma=0.0, **solver_args) for rhs in product_vecs]
                    else:
                        response = [solve_response(matrix, RV(rhs), -k[2], gamma=-k[3], **solver_args) for rhs in product_vecs]
                    for vv in v:
                        response_dict[vv] = np.array(response, dtype=object)
                else:
                    raise ValueError()

            else:
                raise ValueError()
    
    if rvecs_dict_list:
        root_expr = rvecs_dict_list[-1][0]
    else:
        root_expr = mod_isr

    dtype = float
    if gamma_val != 0.0:
        dtype = complex
    res_tens = np.zeros((3,)*len(sos.operators), dtype=dtype)
    
    if isinstance(root_expr, Add):
        term_list = [arg for arg in root_expr.args]
    else:
        term_list = [root_expr]
    
    if symmetric:
        components = list(combinations_with_replacement([0, 1, 2], len(sos.operators))) # if tensor is symmetric
    else:
        components = list(product([0, 1, 2], repeat=len(sos.operators)))
    for c in components:
        comp_map = {
                ABC[ic]: cc for ic, cc in enumerate(c)
        }
        
        #subs_dict = {o[0]: o[1] for o in all_omegas}
        #subs_dict[gamma] = gamma_val
        
        for term in term_list:
            subs_dict = {o[0]: o[1] for o in all_omegas}
            subs_dict[gamma] = gamma_val
            for i, a in enumerate(term.args):
                oper_a = a
                if isinstance(a, adjoint):
                    oper_a = a.args[0]
                if isinstance(oper_a, MTM):
                    if oper_a.op_type == "electric":
                        mtms = mtms_el
                    else:
                        mtms = mtms_mag
                    lhs = term.args[i-1]
                    rhs = term.args[i+1]
                    if oper_a != a and isinstance(rhs, ResponseVector): # Dagger(F) * X
                        subs_dict[a*rhs] = from_vec_to_vec(
                                mtms[comp_map[oper_a.comp]], response_dict[rhs.no][comp_map[rhs.comp]]
                        )
                    elif oper_a == a and isinstance(lhs.args[0], ResponseVector): # Dagger(X) * F
                        subs_dict[lhs*oper_a] = from_vec_to_vec(
                                response_dict[lhs.args[0].no][comp_map[lhs.args[0].comp]], mtms[comp_map[oper_a.comp]]
                        )
                    else:
                        raise ValueError("MTM cannot be evaluated.")
                elif isinstance(a, S2S_MTM): # from_vec * B * to_vec --> transition polarizability
                    if a.op_type == "electric":
                        dips = dips_el
                    else:
                        dips = dips_mag
                    from_v = term.args[i-1]
                    to_v = term.args[i+1]
                    key = from_v*a*to_v
                    if isinstance(from_v, Bra): # <f| B * to_vec
                        fv = state.excitation_vector[final_state[1]]
                    elif isinstance(from_v.args[0], ResponseVector): # Dagger(X) * B * to_vec
                        comp_list_int = [comp_map[char] for char in list(from_v.args[0].comp)]
                        fv = response_dict[from_v.args[0].no][tuple(comp_list_int)]
                    else:
                        raise ValueError("Transition polarizability cannot be evaluated.")
                    if isinstance(to_v, Ket): # from_vec * B |f> 
                        tv = state.excitation_vector[final_state[1]]
                    elif isinstance(to_v, ResponseVector): # from_vec * B * X
                        comp_list_int = [comp_map[char] for char in list(to_v.comp)]
                        tv = response_dict[to_v.no][tuple(comp_list_int)]
                    else:
                        raise ValueError("Transition polarizability cannot be evaluated.")
                    if isinstance(fv, AmplitudeVector) and isinstance(tv, AmplitudeVector):
                        subs_dict[key] = transition_polarizability(
                                property_method, mp, fv, dips[comp_map[a.comp]], tv
                        )
                    else:
                        if isinstance(fv, AmplitudeVector):
                            fv = RV(fv)
                        elif isinstance(tv, AmplitudeVector):
                            tv = RV(tv)
                        subs_dict[key] = transition_polarizability_complex(
                                property_method, mp, fv, dips[comp_map[a.comp]], tv
                        )
                elif isinstance(a, DipoleMoment):
                    if a.from_state == "0" and a.to_state == "0":
                        if a.op_type == "electric":
                            gs_dip_moment = mp.dipole_moment(property_method.level)
                        else:
                            gs_dip_moment = gs_magnetic_dipole_moment(mp, property_method.level)
                        subs_dict[a] = gs_dip_moment[comp_map[a.comp]]
                    elif a.from_state == "0" and a.to_state == str(final_state[0]):
                        if a.op_type == "electric":
                            subs_dict[a] = state.transition_dipole_moment[final_state[1]][comp_map[a.comp]]
                        else:
                            subs_dict[a] = state.transition_magnetic_dipole_moment[final_state[1]][comp_map[a.comp]]
                    else:
                        raise ValueError("Unknown dipole moment.")
                elif isinstance(a, LeviCivita):
                    subs_dict[a] = lc_tensor[c]
            res = term.subs(subs_dict)
            if res == zoo:
                raise ZeroDivisionError()
            res_tens[c] += res
        #print(root_expr, subs_dict)
        #res = root_expr.subs(subs_dict)
        #print(res)
        #if res == zoo:
        #    raise ZeroDivisionError()
        #res_tens[c] = res
        if symmetric:
            perms = list(permutations(c)) # if tensor is symmetric
            for p in perms:
                res_tens[p] = res_tens[c]
    return res_tens


def evaluate_property_sos(
        state, sos_expr, summation_indices, omegas=None, gamma_val=0.0,
        final_state=None, perm_pairs=None, extra_terms=True, symmetric=False, excluded_cases=None
    ):
    """Compute a molecular property from its SOS expression.

    Parameters
    ----------
    state: <class 'adcc.ExcitedStates.ExcitedStates'>
        ExcitedStates object returned by an ADC calculation that includes all states of the system.

    sos_expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
        SymPy expression of the SOS;
        it can be either the full expression or a single term from which the full expression can be generated via permutation.

    summation_indices: list of <class 'sympy.core.symbol.Symbol'>
        List of indices of summation.

    omegas: list of tuples, optional
        List of (symbol, value) pairs for the frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'> or <class 'sympy.core.symbol.Symbol'> or float),
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
    
    correlation_btw_freq = [tup for tup in omegas if type(tup[1]) == Symbol or type(tup[1]) == Add]
    all_omegas = omegas.copy()
    if final_state:
        assert type(final_state) == tuple and len(final_state) == 2
        all_omegas.append(
                (TransitionFrequency(str(final_state[0]), real=True),
                state.excitation_energy_uncorrected[final_state[1]])
        )
    else:
        assert final_state is None
    sos = SumOverStates(
            sos_expr, summation_indices, correlation_btw_freq=correlation_btw_freq, perm_pairs=perm_pairs, excluded_cases=excluded_cases
    )
    _check_omegas_and_final_state(sos.expr, omegas, sos.correlation_btw_freq, gamma_val, final_state)
    
    # all terms are stored as dictionaries in a list
    if isinstance(sos.expr, Add):
        term_list = [
                {"expr": term, "summation_indices": sos.summation_indices, "transition_frequencies": sos.transition_frequencies}
                for term in sos.expr.args
        ]
    else:
        term_list = [
                {"expr": sos.expr, "summation_indices": sos.summation_indices, "transition_frequencies": sos.transition_frequencies}
        ]
    if extra_terms:
        ets = compute_extra_terms(
                sos.expr, sos.summation_indices, excluded_cases=sos.excluded_cases, correlation_btw_freq=sos.correlation_btw_freq
        )
        if isinstance(ets, Add):
            et_list = list(ets.args)
        elif isinstance(ets, Mul):
            et_list = [ets]
        else:
            et_list = []
        for et in et_list:
            sum_ind = find_indices(et, sos.summation_indices) # the extra terms contain less indices of summation
            trans_freq = [TransitionFrequency(str(index), real=True) for index in sum_ind]
            term_list.append(
                    {"expr": et, "summation_indices": sum_ind, "transition_frequencies": trans_freq}
            )
    
    dtype = float
    if gamma_val != 0.0:
        dtype = complex
    res_tens = np.zeros((3,)*len(sos.operators), dtype=dtype)
    
    if isinstance(state, MockExcitedStates):
        pm_level = state.property_method.replace("adc", "")
    else:
        pm_level = state.property_method.level
    if "electric" in set([op.op_type for op in sos.operators]):
        tdms_el = state.transition_dipole_moment
    if "magnetic" in set([op.op_type for op in sos.operators]):
        tdms_mag = state.transition_magnetic_dipole_moment
    s2s_tdms_el = None # state-to-state transition moments are calculated below if needed
    s2s_tdms_f_el = None
    s2s_tdms_mag = None
    s2s_tdms_f_mag = None

    if symmetric:
        components = list(combinations_with_replacement([0, 1, 2], len(sos.operators))) # if tensor is symmetric
    else:
        components = list(product([0, 1, 2], repeat=len(sos.operators)))
    
    for term_dict in tqdm(term_list):
        mod_expr = replace_bra_op_ket(
                term_dict["expr"].subs(sos.correlation_btw_freq)
        )
        sum_ind_str = [str(si) for si in term_dict["summation_indices"]]
        
        # values that the indices of summation can take on
        indices = list(
                product(range(len(state.excitation_energy_uncorrected)), repeat=len(term_dict["summation_indices"]))
        )
        dip_mom_list = [a for a in mod_expr.args if isinstance(a, DipoleMoment)]
        lc_contained = False
        for a in mod_expr.args:
            if isinstance(a, LeviCivita):
                lc_contained = True
        for i in indices:
            state_map = {
                    sum_ind_str[ii]: ind for ii, ind in enumerate(i)
                }
            if final_state:
                state_map[str(final_state[0])] = final_state[1]
            for c in components:
                comp_map = {
                        ABC[ic]: cc for ic, cc in enumerate(c)
                }
                subs_dict = {o[0]: o[1] for o in all_omegas}
                subs_dict[gamma] = gamma_val
                
                for si, tf in zip(sum_ind_str, term_dict["transition_frequencies"]):
                    subs_dict[tf] = state.excitation_energy_uncorrected[state_map[si]]

                for a in dip_mom_list:
                    if a.from_state == "0" and a.to_state == "0":
                        if a.op_type == "electric":
                            if isinstance(state, MockExcitedStates):
                                gs_dip_moment = state.ground_state.dipole_moment[pm_level]
                            else:
                                gs_dip_moment = state.ground_state.dipole_moment(pm_level)
                        else:
                            gs_dip_moment = gs_magnetic_dipole_moment(state.ground_state, pm_level)
                        subs_dict[a] = gs_dip_moment[comp_map[a.comp]]
                    elif a.from_state == "0":
                        index = state_map[a.to_state]
                        comp = comp_map[a.comp]
                        if a.op_type == "electric":
                            subs_dict[a] = tdms_el[index][comp]
                        else:
                            subs_dict[a] = tdms_mag[index][comp]
                    elif a.to_state == "0":
                        index = state_map[a.from_state]
                        comp = comp_map[a.comp]
                        if a.op_type == "electric":
                            subs_dict[a] = tdms_el[index][comp]
                        else:
                            subs_dict[a] = tdms_mag[index][comp]
                    else:
                        index1 = state_map[a.from_state]
                        index2 = state_map[a.to_state]
                        comp = comp_map[a.comp]
                        if a.op_type == "electric":
                            s2s_tdms = s2s_tdms_el
                            s2s_tdms_f = s2s_tdms_f_el
                        else:
                            s2s_tdms = s2s_tdms_mag
                            s2s_tdms_f = s2s_tdms_f_mag
                        if a.from_state in sum_ind_str and a.to_state in sum_ind_str: # e.g., <n|\mu|m>
                            if s2s_tdms is None:
                                s2s_tdms = state_to_state_transition_moments(state, a.op_type)
                            subs_dict[a] = s2s_tdms[index1, index2, comp]
                        elif a.from_state in sum_ind_str: # e.g., <n|\mu|f>
                            if s2s_tdms_f is None:
                                s2s_tdms_f = state_to_state_transition_moments(state, a.op_type, index2)
                            subs_dict[a] = s2s_tdms_f[index1, comp]
                        elif a.to_state in sum_ind_str: # e.g., <f|\mu|n>
                            if s2s_tdms_f is None:
                                s2s_tdms_f = state_to_state_transition_moments(state, a.op_type, index1)
                            subs_dict[a] = s2s_tdms_f[index2, comp]
                        else:
                            raise ValueError()
                        if a.op_type == "electric":
                            s2s_tdms_el = s2s_tdms
                            s2s_tdms_f_el = s2s_tdms_f
                        else:
                            s2s_tdms_mag = s2s_tdms
                            s2s_tdms_f_mag = s2s_tdms_f
                if lc_contained:
                    subs_dict[LeviCivita()] = lc_tensor[c]
                res = mod_expr.xreplace(subs_dict)
                if res == zoo:
                    raise ZeroDivisionError()
                res_tens[c] += res
                if symmetric:
                    perms = list(permutations(c)) # if tensor is symmetric
                    for p in perms:
                        res_tens[p] = res_tens[c]
    return res_tens


def evaluate_property_sos_fast(
        state, sos_expr, summation_indices, omegas=None, gamma_val=0.0,
        final_state=None, perm_pairs=None, extra_terms=True, excluded_cases=None
    ):
    """Compute a molecular property from its SOS expression using the Einstein summation convention.

    Parameters
    ----------
    state: <class 'adcc.ExcitedStates.ExcitedStates'>
        ExcitedStates object returned by an ADC calculation that includes all states of the system.

    sos_expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
        SymPy expression of the SOS;
        it can be either the full expression or a single term from which the full expression can be generated via permutation.
        It already includes the additional terms.

    summation_indices: list of <class 'sympy.core.symbol.Symbol'>
        List of indices of summation.

    omegas: list of tuples, optional
        List of (symbol, value) pairs for the frequencies;
        (symbol, value): (<class 'sympy.core.symbol.Symbol'>, <class 'sympy.core.add.Add'> or <class 'sympy.core.symbol.Symbol'> or float),
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

    correlation_btw_freq = [tup for tup in omegas if type(tup[1]) == Symbol or type(tup[1]) == Add]
    subs_dict = {om_tup[0]: om_tup[1] for om_tup in omegas}
    if final_state:
        assert type(final_state) == tuple and len(final_state) == 2
        subs_dict[TransitionFrequency(str(final_state[0]), real=True)] = (
            state.excitation_energy_uncorrected[final_state[1]]
        )
    else:
        assert final_state is None
    subs_dict[gamma] = gamma_val
    sos = SumOverStates(
            sos_expr, summation_indices, correlation_btw_freq=correlation_btw_freq, perm_pairs=perm_pairs, excluded_cases=excluded_cases
    )
    _check_omegas_and_final_state(sos.expr, omegas, correlation_btw_freq, gamma_val, final_state)

    if extra_terms:
        sos_with_et = sos.expr + compute_extra_terms(
                sos.expr, sos.summation_indices, excluded_cases=sos.excluded_cases, correlation_btw_freq=sos.correlation_btw_freq
        )
        sos_expr_mod = sos_with_et.subs(correlation_btw_freq)
    else:
        sos_expr_mod = sos.expr.subs(correlation_btw_freq)

    dtype = float
    if gamma_val != 0.0:
        dtype = complex
    res_tens = np.zeros((3,)*len(sos.operators), dtype=dtype)

    if isinstance(sos_expr_mod, Add):
        term_list = [replace_bra_op_ket(arg) for arg in sos_expr_mod.args]
    else:
        term_list = [replace_bra_op_ket(sos_expr_mod)]
    
    if isinstance(state, MockExcitedStates):
        pm_level = state.property_method.replace("adc", "")
    else:
        pm_level = state.property_method.level
    s2s_tdms_el = None # state-to-state transition moments are calculated below if needed
    s2s_tdms_f_el = None
    s2s_tdms_mag = None
    s2s_tdms_f_mag = None

    for term in term_list:
        einsum_list = []
        sign = 1
        for a in term.args:
            if isinstance(a, DipoleMoment):
                print(a.op_type)
                if a.from_state == "0" and a.to_state == "0": # <0|\mu|0>
                    if a.op_type == "electric":
                        if isinstance(state, MockExcitedStates):
                            gs_dip_moment = state.ground_state.dipole_moment[pm_level]
                        else:
                            gs_dip_moment = state.ground_state.dipole_moment(pm_level)
                    else:
                        gs_dip_moment = gs_magnetic_dipole_moment(state.ground_state, pm_level)
                    einsum_list.append(("", a.comp, gs_dip_moment))
                elif a.from_state == "0":
                    if a.op_type == "electric":
                        tdms = state.transition_dipole_moment
                    else:
                        tdms = state.transition_magnetic_dipole_moment
                    if a.to_state in sos.summation_indices_str: # e.g., <0|\mu|n>
                        einsum_list.append((a.to_state, a.comp, tdms))
                    else: # e.g., <0|\mu|f>
                        einsum_list.append(("", a.comp, tdms[final_state[1]]))
                elif a.to_state == "0":
                    if a.op_type == "electric":
                        tdms = state.transition_dipole_moment
                    else:
                        tdms = state.transition_magnetic_dipole_moment
                    if a.from_state in sos.summation_indices_str: # e.g., <n|\mu|0>
                        einsum_list.append((a.from_state, a.comp, tdms))
                    else: # e.g., <f|\mu|0>
                        einsum_list.append(("", a.comp, tdms[final_state[1]]))
                else:
                    if a.op_type == "electric":
                        s2s_tdms = s2s_tdms_el
                        s2s_tdms_f = s2s_tdms_f_el
                    else:
                        s2s_tdms = s2s_tdms_mag
                        s2s_tdms_f = s2s_tdms_f_mag
                    if a.from_state in sos.summation_indices_str and a.to_state in sos.summation_indices_str: # e.g., <n|\mu|m>
                        if s2s_tdms is None:
                            s2s_tdms = state_to_state_transition_moments(state, a.op_type)
                        einsum_list.append((a.from_state+a.to_state, a.comp, s2s_tdms))
                    elif a.from_state in sos.summation_indices_str and a.to_state == str(final_state[0]): # e.g., <n|\mu|f>
                        if s2s_tdms_f is None:
                            s2s_tdms_f = state_to_state_transition_moments(state, a.op_type, final_state[1])
                        einsum_list.append((a.from_state, a.comp, s2s_tdms_f))
                    elif a.to_state in sos.summation_indices_str and a.from_state == str(final_state[0]): # e.g., <f|\mu|n>
                        if s2s_tdms_f is None:
                            s2s_tdms_f = state_to_state_transition_moments(state, a.op_type, final_state[1])
                        einsum_list.append((a.to_state, a.comp, s2s_tdms_f))
                    else:
                        raise ValueError()
                    if a.op_type == "electric":
                        s2s_tdms_el = s2s_tdms
                        s2s_tdms_f_el = s2s_tdms_f
                    else:
                        s2s_tdms_mag = s2s_tdms
                        s2s_tdms_f_mag = s2s_tdms_f

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
                        raise ZeroDivisionError()
                    einsum_list.append((index, "", array))
            
            elif isinstance(a, LeviCivita):
                einsum_list.append(("", "ABC", lc_tensor))

            elif Integer(a) is S.NegativeOne:
                sign = -1

            else:
                raise ValueError()
        
        einsum_left = ""
        einsum_right = ""
        array_list = []
        # create string of subscript labels and list of np.arrays for np.einsum
        for tup in einsum_list:
            einsum_left += tup[0] + tup[1] + ","
            einsum_right += tup[1]
            array_list.append(tup[2])
        einsum_left_mod = einsum_left[:-1]
        einsum_right_list = list(set(einsum_right))
        einsum_right_list.sort()
        einsum_right_mod = ''.join(einsum_right_list)
        einsum_string = einsum_left_mod + " -> " + einsum_right_mod
        print(einsum_string)
        res_tens += sign * np.einsum(einsum_string, *array_list)
    
    return res_tens


if __name__ == "__main__":
    from pyscf import gto, scf
    import adcc
    from responsefun.testdata import cache
    from adcc.Excitation import Excitation
    from respondo.polarizability import static_polarizability, real_polarizability, complex_polarizability
    from respondo.rixs import rixs_scattering_strength, rixs
    from respondo.tpa import tpa_resonant
    import time
    from responsefun.test_property import SOS_expressions
    from responsefun.mcd_ref import mcd_bterm

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
    #alpha_tens = evaluate_property_isr(state, alpha_term, [n], omega_alpha, gamma_val=gamma_val)
    #print(alpha_tens)
    #alpha_tens_ref = complex_polarizability(refstate, "adc2", 0.5, gamma_val)
    #print(alpha_tens_ref)
    
    gamma_term = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, m) * TransitionMoment(m, op_c, p) * TransitionMoment(p, op_d, O)
            / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    )
    gamma_omegas = [(w_1, 0.5), (w_2, 0.55), (w_3, 0.6), (w_o, w_1+w_2+w_3)]
    #gamma_tens1 = (
    #        evaluate_property_isr(state, gamma_term, [m, n, p], gamma_omegas, extra_terms=False)
    #)
    #print(gamma_tens1)
    #gamma_tens1_sos = (
    #        evaluate_property_sos_fast(mock_state, gamma_term, [m, n, p], gamma_omegas, extra_terms=False)
    #)
    #print(gamma_tens1_sos)
    #np.testing.assert_allclose(gamma_tens1, gamma_tens1_sos, atol=1e-6)

    
    threepa_term = (
            TransitionMoment(O, op_a, m) * TransitionMoment(m, op_b, n) * TransitionMoment(n, op_c, f)
            / ((w_n - w_1 - w_2) * (w_m - w_1))
    )
    threepa_perm_pairs = [(op_a, w_1), (op_b, w_2), (op_c, w_3)]
    threepa_omegas = [
            (w_1, state.excitation_energy[0]/3),
            (w_2, state.excitation_energy[0]/3),
            (w_3, state.excitation_energy[0]/3),
            (w_f, w_1+w_2+w_3)
    ]
    #threepa_tens = (
    #        evaluate_property_isr(state, threepa_term, [m, n], threepa_omegas, perm_pairs=threepa_perm_pairs, final_state=(f, 0))
    #)
    #print(threepa_tens)
    #threepa_tens_sos = (
    #        evaluate_property_sos_fast(state, threepa_term, [m, n], threepa_omegas, perm_pairs=threepa_perm_pairs, final_state=(f, 0))
    #)
    #print(threepa_tens_sos)
    #np.testing.assert_allclose(threepa_tens, threepa_tens_sos, atol=1e-6)

    # TODO: make it work for esp also in the static case --> projecting the fth eigenstate out of the matrix
    omega_alpha = [(w, 0.5)]
    esp_terms = (
        TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - w_f - w - 1j*gamma)
        + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - w_f + w + 1j*gamma)
    )
    #esp_tens = evaluate_property_isr(
    #        state, esp_terms, [n], omega_alpha, 0.0/Hartree, final_state=(f, 0)#, excluded_cases=[(n, f)]
    #)
    #print(esp_tens)
    #esp_tens_sos = evaluate_property_sos_fast(
    #        mock_state, esp_terms, [n], omega_alpha, 0.0/Hartree, final_state=(f, 0)#, excluded_cases=[(n, f)]
    #)
    #print(esp_tens_sos)
    #np.testing.assert_allclose(esp_tens, esp_tens_sos, atol=1e-7)

    epsilon = LeviCivita()
    mcd_term1 = (
            -1.0 * epsilon
            * TransitionMoment(O, opm_b, k) * TransitionMoment(k, op_c, f) * TransitionMoment(f, op_a, O)
            / w_k
    )
    #mcd_tens1 = evaluate_property_isr(
    #        state, mcd_term1, [k], final_state=(f, 0), extra_terms=False
    #)
    #print(mcd_tens1)
    #mcd_tens1_sos = evaluate_property_sos_fast(
    #        mock_state, mcd_term1, [k], final_state=(f, 0), extra_terms=False
    #)
    #print(mcd_tens1_sos)
    #np.testing.assert_allclose(mcd_tens1, mcd_tens1_sos, atol=1e-7)
    #mcd_term2 = (
    #        -1.0 * epsilon
    #        * TransitionMoment(O, op_c, k) * TransitionMoment(k, opm_b, f) * TransitionMoment(f, op_a, O)
    #        / (w_k - w_f)
    #)
    #mcd_tens2 = evaluate_property_isr(
    #        state, mcd_term2, [k], final_state=(f, 0), extra_terms=False
    #)
    #print(mcd_tens2)
    #print(mcd_tens1-mcd_tens2)
    
    #excited_state = Excitation(state, 0, "adc2")
    #mcd_ref = mcd_bterm(excited_state)
    #print(mcd_ref)
