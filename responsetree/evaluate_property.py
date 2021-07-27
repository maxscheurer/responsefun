import numpy as np
import string

from sympy.physics.quantum.state import Bra, Ket, StateBase
from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex, im
from itertools import permutations, product, combinations_with_replacement

from responsetree.symbols_and_labels import *
from responsetree.response_operators import MTM, S2S_MTM, ResponseVector, Matrix, DipoleOperator, DipoleMoment
from responsetree.sum_over_states import TransitionMoment, SumOverStates
from responsetree.isr_conversion import to_isr
from responsetree.build_tree import build_tree

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


Hartree = 27.211386
ABC = list(string.ascii_uppercase)


def state_to_state_transition_moments(state):
    dips = state.reference_state.operators.electric_dipole
    s2s_tdms = np.zeros((state.size, state.size, 3))
    for ee1 in state.excitations:
        i = ee1.index
        for ee2 in state.excitations:
            j = ee2.index
            tdm = state2state_transition_dm(
                state.property_method,
                state.ground_state,
                ee1.excitation_vector,
                ee2.excitation_vector,
                state.matrix.intermediates,
            )
            tdm_fn = np.array([product_trace(tdm, dip) for dip in dips])
            s2s_tdms[i, j] = tdm_fn
    return s2s_tdms


def from_vec_to_vec(from_vec, to_vec):
    if isinstance(from_vec, AmplitudeVector) and isinstance(to_vec, AmplitudeVector):
        return from_vec @ to_vec
    elif isinstance(from_vec, RV) and isinstance(to_vec, AmplitudeVector):
        rea = from_vec.real @ to_vec
        ima = from_vec.imag @ to_vec
        return rea + 1j*ima
    elif isinstance(from_vec, AmplitudeVector) and isinstance(to_vec, RV):
        rea = from_vec @ to_vec.real
        ima = from_vec @ to_vec.imag
        return rea + 1j*ima
    else:
        raise ValueError()


def evaluate_property_isr(
        state, sos_expr, summation_indices, omegas=None, gamma_val=0.0,
        final_state=None, perm_pairs=None, extra_terms=True, symmetric=False
    ):
    matrix = construct_adcmatrix(state.matrix)
    property_method = select_property_method(matrix)
    mp = matrix.ground_state
    dips = state.reference_state.operators.electric_dipole
    rhss = modified_transition_moments(property_method, mp, dips)
    gs_dip_moment = mp.dipole_moment(property_method.level)

    if omegas is None:
        omegas = []
    if final_state is not None:
        omegas.append(
                (Symbol("w_{{{}}}".format(final_state[0]), real=True),
                state.excitation_energy_uncorrected[final_state[1]])
        )
    correlation_btw_freq = [tup for tup in omegas if type(tup[1]) == Symbol or type(tup[1]) == Add]
    sos = SumOverStates(sos_expr, summation_indices, correlation_btw_freq, perm_pairs)
    isr = to_isr(sos, extra_terms)
    mod_isr = isr.subs(correlation_btw_freq)
    root_expr, rvecs_dict = build_tree(mod_isr)
    
    # check if response equations become equal after inserting values for omegas and gamma
    rvecs_dict_mod = {}
    for k, v in rvecs_dict.items():
        om = float(k[1].subs(omegas))
        gam = float(im(k[2].subs(gamma, gamma_val)))
        if gam == 0 and gamma_val != 0:
            raise ValueError(
                    "Although the entered SOS expression is real, a value for gamma was specified."
            )
        new_key = (k[0], om, gam)
        if new_key not in rvecs_dict_mod.keys():
            rvecs_dict_mod[new_key] = [vv for vv in v.values()]
        else:
            rvecs_dict_mod[new_key] += [vv for vv in v.values()]
    
    # solve response equations
    response_dict = {}
    for k, v in rvecs_dict_mod.items():
        if k[0] == MTM:
            if k[2] == 0.0:
                response = [solve_response(matrix, rhs, -k[1], gamma=0.0) for rhs in rhss]
            else:
                response = [solve_response(matrix, RV(rhs), -k[1], gamma=-k[2]) for rhs in rhss]
            for vv in v:
                response_dict[vv] = response
        else:
            raise ValueError()
    
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
        
        subs_dict = {}
        for o in omegas:
            subs_dict[o[0]] = o[1]
        subs_dict[gamma] = gamma_val
        
        for term in term_list:
            for i, a in enumerate(term.args):
                oper_a = a
                if isinstance(a, adjoint):
                    oper_a = a.args[0]
                if isinstance(oper_a, MTM):
                    lhs = term.args[i-1]
                    rhs = term.args[i+1]
                    if oper_a != a and isinstance(rhs, ResponseVector): # Dagger(F) * X
                        subs_dict[a*rhs] = from_vec_to_vec(
                                rhss[comp_map[oper_a.comp]], response_dict[rhs][comp_map[rhs.comp]]
                        )
                    elif oper_a == a and isinstance(lhs.args[0], ResponseVector): # Dagger(X) * F
                        subs_dict[lhs*oper_a] = from_vec_to_vec(
                                response_dict[lhs.args[0]][comp_map[lhs.args[0].comp]], rhss[comp_map[oper_a.comp]]
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
                        fv = response_dict[from_v.args[0]][comp_map[from_v.args[0].comp]]
                    else:
                        raise ValueError()
                    if isinstance(to_v, Ket): # from_vec * B |f> 
                        tv = state.excitation_vector[final_state[1]]
                    elif isinstance(to_v, ResponseVector): # from_vec * B * X
                        tv = response_dict[to_v][comp_map[to_v.comp]]
                    else:
                        raise ValueError()
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
                        raise ValueError()
        
        res_tens[c] = root_expr.subs(subs_dict)
        if symmetric:
            perms = list(permutations(c)) # if tensor is symmetric
            for p in perms:
                res_tens[p] = res_tens[c]
    return res_tens


def evaluate_property_sos(
        state, sos_expr, summation_indices, omegas=None, gamma_val=0.0,
        final_state=None, gs_terms=None, perm_pairs=None, symmetric=False
    ):
    if omegas is None:
        omegas = []
    if final_state is not None:
        omegas.append(
                (Symbol("w_{{{}}}".format(final_state[0]), real=True),
                state.excitation_energy_uncorrected[final_state[1]])
        )
    correlation_btw_freq = [tup for tup in omegas if type(tup[1]) == Symbol or type(tup[1]) == Add]
    sos = SumOverStates(sos_expr, summation_indices, correlation_btw_freq, perm_pairs)
    sos_expr_mod = sos.expr.subs(correlation_btw_freq)
    
    dtype = float
    if gamma_val != 0.0:
        dtype = complex
    res_tens = np.zeros((3,)*len(sos.operators), dtype=dtype)

    if isinstance(sos_expr_mod, Add):
        term_list = [arg for arg in sos_expr_mod.args]
    else:
        term_list = [sos_expr_mod]

    property_method = state.property_method
    gs_dip_moment = state.ground_state.dipole_moment(property_method.level)
    tdms = state.transition_dipole_moment
    s2s_tdms = None # state-to-state transition moments are calculated below if needed
    
    indices = list(
            product(range(len(state.excitation_energy_uncorrected)), repeat=len(sos.summation_indices))
    )
    if symmetric:
        components = list(combinations_with_replacement([0, 1, 2], len(sos.operators))) # if tensor is symmetric
    else:
        components = list(product([0, 1, 2], repeat=len(sos.operators)))
    for i in indices:
        state_map = {
                sos.summation_indices[ii]: ind for ii, ind in enumerate(i)
            }
        if final_state:
            state_map[final_state[0]] = final_state[1]
        for c in components:
            comp_map = {
                    ABC[ic]: cc for ic, cc in enumerate(c)
            }
            subs_dict = {}
            for o in omegas:
                subs_dict[o[0]] = o[1]
            subs_dict[gamma] = gamma_val
            for si, tf in zip(sos.summation_indices, sos.transition_frequencies):
                subs_dict[tf] = state.excitation_energy_uncorrected[state_map[si]]

            for term in term_list:
                for ia, a in enumerate(term.args):
                    if isinstance(a, DipoleOperator):
                        from_state = term.args[ia-1]
                        to_state = term.args[ia+1]
                        key = from_state*a*to_state
                        if from_state.label[0] == O and to_state.label[0] == O:
                            subs_dict[key] = gs_dip_moment[comp_map[a.comp]]
                        elif from_state.label[0] == O:
                            index = state_map[to_state.label[0]]
                            comp = comp_map[a.comp]
                            subs_dict[key] = tdms[index][comp]
                        elif to_state.label[0] == O:
                            index = state_map[from_state.label[0]]
                            comp = comp_map[a.comp]
                            subs_dict[key] = tdms[index][comp]
                        else:
                            if s2s_tdms is None:
                                s2s_tdms = state_to_state_transition_moments(state)
                            index1 = state_map[from_state.label[0]]
                            index2 = state_map[to_state.label[0]]
                            comp = comp_map[a.comp]
                            subs_dict[key] = s2s_tdms[index1, index2, comp]
            res_tens[c] += sos_expr_mod.subs(subs_dict)
            if symmetric:
                perms = list(permutations(c)) # if tensor is symmetric
                for p in perms:
                    res_tens[p] = res_tens[c]
    
    if gs_terms:
        gs_terms_mod = gs_terms.subs(correlation_btw_freq)
        if isinstance(gs_terms_mod, Add):
            gs_terms_list = [arg for arg in gs_terms_mod.args]
        elif isinstance(gs_terms_mod, Mul):
            gs_terms_list = [gs_terms_mod]
        else:
            raise TypeError("gs_terms must be either of type Mul or Add.")
        for c in components:
            comp_map = {
                    ABC[ic]: cc for ic, cc in enumerate(c)
            }
            subs_dict = {}
            for o in omegas:
                subs_dict[o[0]] = o[1]
            subs_dict[gamma] = gamma_val
            for term in gs_terms_list:
                for ia, a in enumerate(term.args):
                    if isinstance(a, DipoleOperator):
                        from_state = term.args[ia-1]
                        to_state = term.args[ia+1]
                        key = from_state*a*to_state
                        if from_state.label[0] == O and to_state.label[0] == O:
                            subs_dict[key] = gs_dip_moment[comp_map[a.comp]]
                        elif from_state.label[0] == O and to_state.label[0] == final_state[0]:
                            comp = comp_map[a.comp]
                            subs_dict[key] = state.transition_dipole_moment[final_state[1]][comp]
                        elif to_state.label[0] == O and from_state.label[0] == final_state[0]:
                            comp = comp_map[a.comp]
                            subs_dict[key] = state.transition_dipole_moment[final_state[1]][comp]
                        else:
                            raise ValueError()
            res_tens[c] += gs_terms.subs(subs_dict)
            if symmetric:
                perms = list(permutations(c)) # if tensor is symmetric
                for p in perms:
                    res_tens[p] = res_tens[c]
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
    state = adcc.adc2(scfres, n_singlets=65)
    
    omega_alpha = [(w, 0.59)]
    alpha_terms = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
            + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w + 1j*gamma)
    )
    alpha_tens = evaluate_property_isr(state, alpha_terms, [n], omega_alpha, gamma_val=0.124/Hartree, symmetric=True)
    print(alpha_tens)
    alpha_ref = complex_polarizability(matrix, omega=omega_alpha[0][1], gamma=0.124/Hartree)
    print(alpha_ref)
    np.testing.assert_allclose(alpha_tens, alpha_ref, atol=1e-7)
    alpha_tens_sos = evaluate_property_sos(state, alpha_terms, [n], omega_alpha, gamma_val=0.124/Hartree, symmetric=True)
    print(alpha_tens_sos)
    np.testing.assert_allclose(alpha_tens, alpha_tens_sos, atol=1e-7)

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
    #gs_terms_rot_wav = (
    #        (TransitionMoment(O, op_a, f) * TransitionMoment(O, op_b, O) / (-w-1j*gamma))
    #        - (TransitionMoment(O, op_b, f) * TransitionMoment(O, op_a, O) / (w-w_f+1j*gamma))
    #)
    #rixs_tens_sos = evaluate_property_sos(state, rixs_term_short, [n], omega_rixs, gamma_val=0.124/Hartree, final_state=(f, 0), gs_terms=gs_terms_rot_wav)
    #print(rixs_tens_sos)
    #np.testing.assert_allclose(rixs_tens, rixs_tens_sos, atol=1e-7)

    omegas_beta = [(w_1, 0.5), (w_2, 0.5), (w_o, w_1+w_2)]
    beta_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o - 1j*gamma) * (w_k - w_2 - 1j*gamma))
    #beta_tens = evaluate_property_isr(
    #        state, beta_term, [n, k], omegas_beta, gamma_val=0.01,
    #        perm_pairs=[(op_a, -w_o-1j*gamma), (op_b, w_1+1j*gamma), (op_c, w_2+1j*gamma)], extra_terms=False
    #)
    #print(beta_tens)
    #print(evaluate_property_sos(
    #    state, beta_term, [n, k], omegas_beta, gamma_val=0.01,
    #    perm_pairs=[(op_a, -w_o-1j*gamma), (op_b, w_1+1j*gamma), (op_c, w_2+1j*gamma)]
    #))

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
    #print(evaluate_property_sos(state, tpa_terms, [n], final_state=(f, 0)))
    
    omegas_gamma = [(w_1, 0.5), (w_2, 0.3), (w_3, 0.0), (w_o, w_1+w_2+w_3)]
    gamma_extra_terms = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) * TransitionMoment(O, op_c, m) * TransitionMoment(m, op_d, O)
            / ((w_n - w_o) * (w_m - w_3) * (w_m + w_2))
    )
    #gamma_et_tens = evaluate_property_sos(
    #        state, gamma_extra_terms, [n, m], omegas_gamma, perm_pairs=[(op_a, -w_o), (op_b, w_1), (op_c, w_2), (op_d, w_3)]
    #)
    #print(gamma_et_tens)
    
    # TODO: make it work for esp
    esp_terms = (
        TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - w_f - w - 1j*gamma)
        + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - w_f + w + 1j*gamma)
    )
    #esp_tens = evaluate_property_isr(state, esp_terms, [n], omega_alpha, gamma_val=0.124/Hartree, final_state=(f, 0))
    #print(esp_tens)
