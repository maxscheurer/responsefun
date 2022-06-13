import numpy as np
import string
from scipy.constants import physical_constants

from sympy.physics.quantum.state import Bra, Ket, StateBase
from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex, im, Float, Integer, S, zoo, I
from itertools import permutations, product, combinations_with_replacement

from responsefun.symbols_and_labels import *
from responsefun.response_operators import MTM, S2S_MTM, ResponseVector, DipoleOperator, DipoleMoment, TransitionFrequency
from responsefun.sum_over_states import TransitionMoment, SumOverStates
from responsefun.isr_conversion import to_isr, compute_extra_terms
from responsefun.build_tree import build_tree
from responsefun.testdata import cache
from responsefun.testdata.cache import MockExcitedStates
from responsefun.b_matrix_vector_product import b_matrix_vector_product

from pyscf import gto, scf
import adcc
from adcc import AmplitudeVector
from adcc.workflow import construct_adcmatrix
from adcc.adc_pp import modified_transition_moments
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.Excitation import Excitation
from adcc.State2States import State2States
from adcc.OneParticleOperator import product_trace
from respondo.misc import select_property_method
from respondo.solve_response import solve_response, transition_polarizability, transition_polarizability_complex
from respondo.polarizability import static_polarizability, real_polarizability, complex_polarizability
from respondo.cpp_algebra import ResponseVector as RV
from respondo.rixs import rixs_scattering_strength, rixs
from respondo.tpa import tpa_resonant
import time
from tqdm import tqdm


Hartree = physical_constants["hartree-electron volt relationship"][0]
ABC = list(string.ascii_uppercase)


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
            subs_dict[key] = DipoleMoment(a.comp, str(from_state.label[0]), str(to_state.label[0]))
    return expr.subs(subs_dict)


def state_to_state_transition_moments(state, final_state=None):
    if isinstance(state, MockExcitedStates):
        if final_state is None:
            return state.transition_dipole_moment_s2s
        else:
            return state.transition_dipole_moment_s2s[:, final_state]
    else:
        dips = state.reference_state.operators.electric_dipole
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
                tdm_fn = np.array([product_trace(tdm, dip) for dip in dips])
                s2s_tdms[i, j] = tdm_fn
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
    dips = state.reference_state.operators.electric_dipole
    mtms = modified_transition_moments(property_method, mp, dips)
    gs_dip_moment = mp.dipole_moment(property_method.level)

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
            om = float(k[1].subs(all_omegas))
            gam = float(im(k[2].subs(gamma, gamma_val)))
            if gam == 0 and gamma_val != 0:
                raise ValueError(
                        "Although the entered SOS expression is real, a value for gamma was specified."
                )
            new_key = (k[0], om, gam)
            if new_key not in rvecs_dict_mod.keys():
                rvecs_dict_mod[new_key] = [v]
            else:
                rvecs_dict_mod[new_key].append(v)
        
        # solve response equations
        for k, v in rvecs_dict_mod.items():
            if k[0] == MTM:
                if k[2] == 0.0:
                    response = [solve_response(matrix, rhs, -k[1], gamma=0.0, **solver_args) for rhs in mtms]
                else:
                    response = [solve_response(matrix, RV(rhs), -k[1], gamma=-k[2], **solver_args) for rhs in mtms]
                for vv in v:
                    response_dict[vv] = np.array(response, dtype=object)
            elif type(k[0]) == tuple:
                if len(k[0]) == 3:
                    if k[2] == 0.0:
                        no = k[0][2]
                        #print("no rvec: ", no)
                        rvecs = response_dict[no]
                        product_vecs = b_matrix_vector_product(property_method, mp, dips, rvecs)
                        iterables = [list(range(shape)) for shape in product_vecs.shape]
                        components = list(product(*iterables))
                        response = np.empty(product_vecs.shape, dtype=object)
                        for c in components:
                            rhs = product_vecs[c]
                            response[c] = solve_response(matrix, rhs, -k[1], gamma=0.0)
                    else:
                        raise NotImplementedError("The case of complex response vectors has not been implemented yet.")
                    for vv in v:
                        response_dict[vv] = response
                elif len(k[0]) == 2:
                    product_vecs = b_matrix_vector_product(property_method, mp, dips, state.excitation_vector[final_state[1]])
                    if k[2] == 0.0:
                        response = [solve_response(matrix, rhs, -k[1], gamma=0.0, **solver_args) for rhs in product_vecs]
                    else:
                        response = [solve_response(matrix, RV(rhs), -k[1], gamma=-k[2], **solver_args) for rhs in product_vecs]
                    for vv in v:
                        response_dict[vv] = np.array(response, dtype=object)
                else:
                    raise NotImplementedError()

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
                        subs_dict[a] = gs_dip_moment[comp_map[a.comp]]
                    elif a.from_state == "0" and a.to_state == str(final_state[0]):
                        subs_dict[a] = state.transition_dipole_moment[final_state[1]][comp_map[a.comp]] 
                    else:
                        raise ValueError("Unknown dipole moment.")
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
        ets = compute_extra_terms(sos.expr, sos.summation_indices, correlation_btw_freq=sos.correlation_btw_freq)
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
        pm = state.property_method.replace("adc", "")
        gs_dip_moment = state.ground_state.dipole_moment[pm]
    else:
        gs_dip_moment = state.ground_state.dipole_moment(state.property_method.level)
    tdms = state.transition_dipole_moment
    s2s_tdms = None # state-to-state transition moments are calculated below if needed
    s2s_tdms_f = None

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
                        subs_dict[a] = gs_dip_moment[comp_map[a.comp]]
                    elif a.from_state == "0":
                        index = state_map[a.to_state]
                        comp = comp_map[a.comp]
                        subs_dict[a] = tdms[index][comp]
                    elif a.to_state == "0":
                        index = state_map[a.from_state]
                        comp = comp_map[a.comp]
                        subs_dict[a] = tdms[index][comp]
                    else:
                        index1 = state_map[a.from_state]
                        index2 = state_map[a.to_state]
                        comp = comp_map[a.comp]
                        if a.from_state in sum_ind_str and a.to_state in sum_ind_str: # e.g., <n|\mu|m>
                            if s2s_tdms is None:
                                s2s_tdms = state_to_state_transition_moments(state)
                            subs_dict[a] = s2s_tdms[index1, index2, comp]
                        elif a.from_state in sum_ind_str: # e.g., <n|\mu|f>
                            if s2s_tdms_f is None:
                                s2s_tdms_f = state_to_state_transition_moments(state, index2)
                            subs_dict[a] = s2s_tdms_f[index1, comp]
                        elif a.to_state in sum_ind_str: # e.g., <f|\mu|n>
                            if s2s_tdms_f is None:
                                s2s_tdms_f = state_to_state_transition_moments(state, index1)
                            subs_dict[a] = s2s_tdms_f[index2, comp]
                        else:
                            raise ValueError()
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
        sos_with_et = sos.expr + compute_extra_terms(sos.expr, sos.summation_indices, correlation_btw_freq=sos.correlation_btw_freq)
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
        pm = state.property_method.replace("adc", "")
        gs_dip_moment = state.ground_state.dipole_moment[pm]
    else:
        gs_dip_moment = state.ground_state.dipole_moment(state.property_method.level)
    tdms = state.transition_dipole_moment
    s2s_tdms = None # state-to-state transition moments are calculated below if needed
    s2s_tdms_f = None

    for term in term_list:
        einsum_list = []
        sign = 1
        for a in term.args:
            if isinstance(a, DipoleMoment):
                if a.from_state == "0" and a.to_state == "0": # <0|\mu|0>
                    einsum_list.append(("", a.comp, gs_dip_moment))
                elif a.from_state == "0":
                    if a.to_state in sos.summation_indices_str: # e.g., <0|\mu|n>
                        einsum_list.append((a.to_state, a.comp, tdms))
                    else: # e.g., <0|\mu|f>
                        einsum_list.append(("", a.comp, tdms[final_state[1]]))
                elif a.to_state == "0":
                    if a.from_state in sos.summation_indices_str: # e.g., <0|\mu|n>
                        einsum_list.append((a.from_state, a.comp, tdms))
                    else: # e.g., <0|\mu|f>
                        einsum_list.append(("", a.comp, tdms[final_state[1]]))
                else:
                    if a.from_state in sos.summation_indices_str and a.to_state in sos.summation_indices_str: # e.g., <n|\mu|m>
                        if s2s_tdms is None:
                            s2s_tdms = state_to_state_transition_moments(state)
                        einsum_list.append((a.from_state+a.to_state, a.comp, s2s_tdms))
                    elif a.from_state in sos.summation_indices_str: # e.g., <n|\mu|f>
                        if s2s_tdms_f is None:
                            s2s_tdms_f = state_to_state_transition_moments(state, final_state[1])
                        einsum_list.append((a.from_state, a.comp, s2s_tdms_f))
                    elif a.to_state in sos.summation_indices_str: # e.g., <f|\mu|n>
                        if s2s_tdms_f is None:
                            s2s_tdms_f = state_to_state_transition_moments(state, final_state[1])
                        einsum_list.append((a.to_state, a.comp, s2s_tdms_f))
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
                        raise ZeroDivisionError()
                    einsum_list.append((index, "", array))
            
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
        einsum_string = einsum_left_mod + " -> " + einsum_right
        print(einsum_string)
        res_tens += sign * np.einsum(einsum_string, *array_list)
    
    return res_tens


if __name__ == "__main__":
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
    state = adcc.adc2(scfres, n_singlets=5)
    mock_state = cache.data_fulldiag["h2o_sto3g_adc2"]
   
#------------------------------------------------------
    omega_alpha = (w, 0.59)
    alpha_terms = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
            + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w + 1j*gamma)
    )
    #alpha_tens = evaluate_property_isr(state, alpha_terms, [n], omega_alpha, gamma_val=0.124/Hartree, symmetric=True)
    #print(alpha_tens)
    #alpha_ref = complex_polarizability(matrix, omega=omega_alpha[1], gamma=0.124/Hartree)
    #np.testing.assert_allclose(alpha_tens, alpha_ref, atol=1e-7)
    #alpha_tens_sos = evaluate_property_sos(mock_state, alpha_terms, [n], omega_alpha, gamma_val=0.124/Hartree, symmetric=True)
    #np.testing.assert_allclose(alpha_tens, alpha_tens_sos, atol=1e-7)
    #alpha_tens_sos_2 = evaluate_property_sos_fast(mock_state, alpha_terms, [n], omega_alpha, gamma_val=0.124/Hartree)
    #print(alpha_tens_sos_2)
    #np.testing.assert_allclose(alpha_tens, alpha_tens_sos_2, atol=1e-7)

    omega_rixs = [(w, 534.74/Hartree)]
    rixs_terms = (
            TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
            + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w - w_f + 1j*gamma)
        )
    rixs_term_short = rixs_terms.args[0]
    #rixs_tens = evaluate_property_isr(state, rixs_term_short, [n], omega_rixs, gamma_val=0.124/Hartree, final_state=(f, 0))
    #print(rixs_tens)
    #rixs_strength = rixs_scattering_strength(rixs_tens, omega_rixs[0][1], omega_rixs[0][1]-state.excitation_energy_uncorrected[0])
    #print(rixs_strength)
    #excited_state = Excitation(state, 0, "adc2")
    #rixs_ref = rixs(excited_state, omega_rixs[0][1], gamma=0.124/Hartree)
    #print(rixs_ref)
    #np.testing.assert_allclose(rixs_tens, rixs_ref[1], atol=1e-7)
    #rixs_tens_sos = evaluate_property_sos(state, rixs_term_short, [n], omega_rixs, gamma_val=0.124/Hartree, final_state=(f, 3))
    #print(rixs_tens_sos)
    #np.testing.assert_allclose(rixs_tens, rixs_tens_sos, atol=1e-7)
    #rixs_tens_sos_2 = evaluate_property_sos_fast(state, rixs_terms, [n], omega_rixs, gamma_val=0.124/Hartree, final_state=(f, 3))
    #print(rixs_tens_sos_2)
    #np.testing.assert_allclose(rixs_tens, rixs_tens_sos_2, atol=1e-7)

    omegas_beta = [(w_1, 0.5), (w_2, 0.5), (w_o, w_1+w_2+1j*gamma)]
    beta_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o - 1j*gamma) * (w_k - w_2 - 1j*gamma))
    #beta_tens = evaluate_property_isr(
    #        state, beta_term, [n, k], omegas_beta, gamma_val=0.01,
    #        perm_pairs=[(op_a, -w_o-1j*gamma), (op_b, w_1+1j*gamma), (op_c, w_2+1j*gamma)]
    #)
    #print(beta_tens)
    #beta_tens_sos = evaluate_property_sos(
    #    state, beta_term, [n, k], omegas_beta,
    #    perm_pairs=[(op_a, -w_o), (op_b, w_1), (op_c, w_2)]
    #)
    #print(beta_tens_sos)
    #np.testing.assert_allclose(beta_tens, beta_tens_sos, atol=1e-7)
    #beta_tens_sos_2 = evaluate_property_sos_fast(
    #    mock_state, beta_term, [n, k], omegas_beta, gamma_val=0.01,
    #    perm_pairs=[(op_a, -w_o-1j*gamma), (op_b, w_1+1j*gamma), (op_c, w_2+1j*gamma)]
    #)
    #print(beta_tens_sos_2)
    #np.testing.assert_allclose(beta_tens, beta_tens_sos_2, atol=1e-7)


    tpa_terms = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - (w_f/2))
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - (w_f/2))
    )
    #tpa_tens = evaluate_property_isr(state, tpa_terms, [n], final_state=(f, 0))
    #print(tpa_tens)
    #excited_state = Excitation(state, 0, "adc2")
    #tpa_ref = tpa_resonant(excited_state)
    #print(tpa_ref)
    #np.testing.assert_allclose(tpa_tens, tpa_ref[1], atol=1e-7)
    #tpa_tens_sos = evaluate_property_sos(state, tpa_terms, [n], final_state=(f, 0)) 
    #print(tpa_tens_sos)
    #np.testing.assert_allclose(tpa_tens, tpa_tens_sos, atol=1e-7)
    #tpa_tens_sos_2 = evaluate_property_sos_fast(state, tpa_terms, [n], final_state=(f, 0))
    #print(tpa_tens_sos_2)
    #np.testing.assert_allclose(tpa_tens, tpa_tens_sos_2, atol=1e-7)

    omegas_gamma = [(w_1, 0.05), (w_2, 0.055), (w_3, 0.06), (w_o, w_1+w_2+w_3)]

    gamma_term = TransitionMoment(O, op_a, n)*TransitionMoment(n, op_b, m) * TransitionMoment(m, op_c, p) * TransitionMoment(p, op_d, O) / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    #gamma_term_tens = evaluate_property_isr(state, gamma_term, [n, m, p], omegas_gamma, extra_terms=False)
    #print(gamma_term_tens)
    #gamma_term_tens_sos = evaluate_property_sos_fast(mock_state, gamma_term, [n, m, p], omegas_gamma, extra_terms=False)
    #print(gamma_term_tens_sos)
    #gamma_term_tens_sos2 = evaluate_property_sos(mock_state, gamma_term, [n, m, p], omegas_gamma, extra_terms=False)
    #print(gamma_term_tens_sos2)
    #np.testing.assert_allclose(gamma_term_tens, gamma_term_tens_sos, atol=1e-7)
    #np.testing.assert_allclose(gamma_term_tens_sos, gamma_term_tens_sos2, atol=1e-7)


    gamma_extra_terms = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) * TransitionMoment(O, op_c, m) * TransitionMoment(m, op_d, O)
            / ((w_n - w_o) * (-w_2 - w_3) * (w_m - w_3))
    )
    #tic = time.perf_counter()
    #gamma_et_tens = evaluate_property_isr(
    #        state, gamma_extra_terms, [n, m], omegas_gamma, perm_pairs=[(op_a, -w_o), (op_b, w_1), (op_c, w_2), (op_d, w_3)], extra_terms=False
    #)
    #toc = time.perf_counter()
    #print(gamma_et_tens)
    #print(toc-tic)
    #gamma_et_tens_sos = evaluate_property_sos_fast(
    #        state, gamma_extra_terms, [n, m], omegas_gamma, perm_pairs=[(op_a, -w_o), (op_b, w_1), (op_c, w_2), (op_d, w_3)]
    #)
    #print(gamma_et_tens_sos)
    #np.testing.assert_allclose(gamma_et_tens_sos, gamma_et_tens_2, atol=1e-7)
    
    gamma_extra_terms_2 = (
            (-1) * TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) * TransitionMoment(O, op_c, m) * TransitionMoment(m, op_d, O)
            / ((w_n - w_o) * (w_m - w_3) * (w_m + w_2))
    )
    #gamma_et_tens = evaluate_property_sos_fast(
    #        state, gamma_extra_terms_2, [n, m], omegas_gamma, perm_pairs=[(op_a, -w_o), (op_b, w_1), (op_c, w_2), (op_d, w_3)], extra_terms=False
    #)
    #print(gamma_et_tens)

    gamma_perm_1 = (
            (-1) * TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) * TransitionMoment(O, op_c, m) * TransitionMoment(m, op_d, O)
            / ((w_n - w_o) * (w_m - w_3) * (w_m + w_2))
    )
    
    gamma_perm_2 = (
            (-1) * TransitionMoment(O, op_c, m) * TransitionMoment(m, op_d, O) * TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O)
            / ((w_m + w_2) * (w_n - w_1) * (w_n - w_o))
    )
    #gamma_perm_1_sos = evaluate_property_sos_fast(
    #        state, gamma_perm_1, [n, m], omegas_gamma, extra_terms=False
    #)
    #gamma_perm_2_sos = evaluate_property_sos_fast(
    #        state, gamma_perm_2, [n, m], omegas_gamma, extra_terms=False
    #)
    #print(gamma_perm_1_sos)
    #print(gamma_perm_2_sos)
    

    #rixs_test1 = TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
    #rixs_test2 = TransitionMoment(O, op_a, O) * TransitionMoment(f, op_b, O) / (w_f - w - 1j*gamma)
    #rixs_test3 = TransitionMoment(O, op_b, O) * TransitionMoment(f, op_a, O) / (-w - 1j*gamma)
    #rixs_test_tensor1 = evaluate_property_isr(state, rixs_test1, [n], omega_rixs, 0.124/Hartree, final_state=(f, 0), extra_terms=False)
    #rixs_test_tensor2 = evaluate_property_isr(state, rixs_test2, [], omega_rixs, 0.124/Hartree, final_state=(f, 0))
    #rixs_test_tensor3 = evaluate_property_isr(state, rixs_test3, [], omega_rixs, 0.124/Hartree, final_state=(f, 0))
    #rixs_test_tensor = rixs_test_tensor1 + rixs_test_tensor2 + rixs_test_tensor3
    #print(rixs_test_tensor)
    #np.testing.assert_allclose(rixs_test_tensor, rixs_tens)

    
    threepa_term = TransitionMoment(O, op_a, m) * TransitionMoment(m, op_b, n) * TransitionMoment(n, op_c, f) / ((w_n - w_1 - w_2) * (w_m - w_1))
    #threepa_tens = evaluate_property_isr(state, threepa_term, [m, n], [(w_1, state.excitation_energy[0]/3), (w_2, state.excitation_energy[0]/3), (w_3, state.excitation_energy[0]/3), (w_f, w_1+w_2+w_3)], perm_pairs=[(op_a, w_1), (op_b, w_2), (op_c, w_3)], final_state=(f, 0))
    #print(threepa_tens)
    #threepa_tens_sos = evaluate_property_sos_fast(mock_state, threepa_term, [m, n], [(w_1, state.excitation_energy[0]/3), (w_2, state.excitation_energy[0]/3), (w_3, state.excitation_energy[0]/3), (w_f, w_1+w_2+w_3)], perm_pairs=[(op_a, w_1), (op_b, w_2), (op_c, w_3)], final_state=(f, 0))
    #print(threepa_tens_sos)
    #np.testing.assert_allclose(threepa_tens, threepa_tens_sos)

    # TODO: make it work for esp
    esp_terms = (
        TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - w_f - w - 1j*gamma)
        #+ TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - w_f + w + 1j*gamma)
    )
    #esp_tens = evaluate_property_isr(state, esp_terms, [n], omega_alpha, final_state=(f, 0), extra_terms=False)
    #print(esp_tens)
    #esp_tens_sos = evaluate_property_sos_fast(mock_state, esp_terms, [n], omega_alpha, final_state=(f, 0), extra_terms=False)
    #print(esp_tens_sos)
    #np.testing.assert_allclose(esp_tens, esp_tens_sos, atol=1e-7)

