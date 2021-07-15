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
#from adcc.workflow import construct_adcmatrix
from adcc import AmplitudeVector
from adcc.adc_pp import modified_transition_moments
from adcc.Excitation import Excitation
from respondo.misc import select_property_method
from respondo.solve_response import solve_response, transition_polarizability, transition_polarizability_complex
from respondo.polarizability import static_polarizability, real_polarizability, complex_polarizability
from respondo.cpp_algebra import ResponseVector as RV
from respondo.rixs import rixs_scattering_strength, rixs
from respondo.tpa import tpa_resonant


Hartree = 27.211386
ABC = list(string.ascii_uppercase)


def from_vec_to_vec(from_vec, to_vec):
    if isinstance(from_vec, AmplitudeVector) and isinstance(to_vec, AmplitudeVector):
        return from_vec @ to_vec
    elif isinstance(from_vec, RV):
        rea = from_vec.real @ to_vec
        ima = from_vec.imag @ to_vec
        return rea + 1j*ima
    elif isinstance(to_vec, RV):
        rea = from_vec @ to_vec.real
        ima = from_vec @ to_vec.imag
        return rea + 1j*ima
    else:
        raise ValueError()


def evaluate_property(
        scfres, method, sos_expr, summation_indices, omegas, gamma_val=0.0,
        final_state=None, perm_pairs=None, extra_terms=True, symmetric=False
    ):
    refstate = adcc.ReferenceState(scfres)
    matrix = adcc.AdcMatrix(method, refstate)
    if final_state:
        state = adcc.run_adc(scfres, method=method, n_singlets=final_state[1]+1)
        omegas.append(
                (Symbol("w_{{{}}}".format(final_state[0]), real=True),
                state.excitation_energy_uncorrected[final_state[1]])
        )
    property_method = select_property_method(matrix)
    mp = matrix.ground_state
    dips = refstate.operators.electric_dipole
    rhss = modified_transition_moments(property_method, mp, dips)
    
    correlation_btw_freq = [tup for tup in omegas if type(tup[1]) == Symbol or type(tup[1]) == Add]
    sos = SumOverStates(sos_expr, summation_indices, correlation_btw_freq, perm_pairs)
    isr = to_isr(sos, extra_terms=extra_terms)
    mod_isr = isr.subs(correlation_btw_freq)
    root_expr, rvecs_dict = build_tree(mod_isr)
    
    dtype = float
    if gamma_val != 0.0:
        dtype = complex
    res_tens = np.zeros((3,)*len(sos.operators), dtype=dtype)
    
    rvecs_dict_mod = {} # check if response equations become equal after inserting values for omegas and gamma
    for k, v in rvecs_dict.items():
        om = float(k[1].subs(omegas))
        gam = float(im(k[2].subs(gamma, gamma_val)))
        if gam == 0 and gamma_val != 0:
            raise ValueError(
                    "Although the entered SOS expression is real, a value for gamma was specified."
            )
        new_key = (k[0], om, gam)
        if new_key not in rvecs_dict_mod.keys():
            rvecs_dict_mod[(k[0], om, gam)] = [vv for vv in v.values()]
        else:
            rvecs_dict_mod[new_key] += [vv for vv in v.values()]
    
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
    
    if isinstance(root_expr, Add):
        term_list = [arg for arg in root_expr.args]
    else:
        term_list = [root_expr]
    
    if symmetric:
        components = list(combinations_with_replacement([0, 1, 2], len(sos.operators))) # if tensor is symmetric
    else:
        components = list(product([0, 1, 2], repeat=len(sos.operators)))
    for c in components:
        index_map = {
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
                                rhss[index_map[oper_a.comp]], response_dict[rhs][index_map[rhs.comp]]
                        )
                    elif oper_a == a and isinstance(lhs.args[0], ResponseVector): # Dagger(X) * F
                        subs_dict[lhs*oper_a] = from_vec_to_vec(
                                response_dict[lhs.args[0]][index_map[lhs.args[0].comp]], rhss[index_map[oper_a.comp]]
                        )
                    else:
                        raise ValueError("MTM cannot be evaluated.")
                elif isinstance(oper_a, S2S_MTM): # from_vec * B * to_vec --> transition polarizability
                    from_v = term.args[i-1]
                    to_v = term.args[i+1]
                    key = from_v*oper_a*to_v
                    if isinstance(from_v, Bra): # <f| B * to_vec
                        fv = state.excitation_vector[final_state[1]]
                    elif isinstance(from_v.args[0], ResponseVector): # Dagger(X) * B * to_vec
                        fv = response_dict[from_v.args[0]][index_map[from_v.args[0].comp]]
                    else:
                        raise ValueError()
                    if isinstance(to_v, Ket): # from_vec * B |f> 
                        tv = state.excitation_vector[final_state[1]]
                    elif isinstance(to_v, ResponseVector): # from_vec * B * X
                        tv = response_dict[to_v][index_map[to_v.comp]]
                    else:
                        raise ValueError()
                    if isinstance(fv, AmplitudeVector) and isinstance(tv, AmplitudeVector):
                        subs_dict[key] = transition_polarizability(
                                property_method, mp, fv, dips[index_map[oper_a.comp]], tv
                        )
                    else:
                        if isinstance(fv, AmplitudeVector):
                            fv = RV(fv)
                        elif isinstance(tv, AmplitudeVector):
                            tv = RV(tv)
                        subs_dict[key] = transition_polarizability_complex(
                                property_method, mp, fv, dips[index_map[oper_a.comp]], tv
                        )
                elif isinstance(oper_a, DipoleMoment):
                    if oper_a.from_state == "0" and oper_a.to_state == "0":
                        subs_dict[oper_a] = mp.dipole_moment(property_method.level)[index_map[oper_a.comp]]
                    elif oper_a.from_state == "0" and oper_a.to_state == str(final_state[0]):
                        subs_dict[oper_a] = state.transition_dipole_moment[final_state[1]][index_map[oper_a.comp]] 
                    else:
                        raise ValueError()
        
        res_tens[c] = root_expr.subs(subs_dict)
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
    state = adcc.adc2(scfres, n_singlets=1)

    omega_alpha = [(w, 0.59)]
    alpha_terms = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
            + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w + 1j*gamma)
    )
    #alpha_tens = evaluate_property(scfres, "adc2", alpha_terms, [n], omega_alpha, gamma_val=0.124/Hartree, symmetric=True)
    #print(alpha_tens)
    #alpha_ref = complex_polarizability(matrix, omega=omega_alpha[0][1], gamma=0.124/Hartree)
    #print(alpha_ref)
    #np.testing.assert_allclose(alpha_tens, alpha_ref, atol=1e-7)

    omega_rixs = [(w, 534.74/Hartree)]
    rixs_terms = (
            TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
            + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, O) / (w_n + w - w_f + 1j*gamma)
        )
    rixs_term_short = rixs_terms.args[0]
    #rixs_tens = evaluate_property(scfres, "adc2", rixs_term_short, [n], omega_rixs, gamma_val=0.124/Hartree, final_state=(f, 0))
    #print(rixs_tens)
    #rixs_strength = rixs_scattering_strength(rixs_tens, omega_rixs[0][1], omega_rixs[0][1]-state.excitation_energy_uncorrected[0])
    #print(rixs_strength)
    #excited_state = Excitation(state, 0, "adc2")
    #rixs_ref = rixs(excited_state, omega_rixs[0][1], gamma=0.124/Hartree)
    #print(rixs_ref)
    #np.testing.assert_allclose(rixs_tens, rixs_ref[1], atol=1e-7)

    omegas_beta = [(w_1, 0.5), (w_2, 0.5), (w_o, w_1+w_2)]
    beta_term = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o - 1j*gamma) * (w_k - w_2 - 1j*gamma))
    #beta_tens = evaluate_property(
    #        scfres, "adc2", beta_term, [n, k], omegas_beta, gamma_val=0.01,
    #        perm_pairs=[(op_a, -w_o-1j*gamma), (op_b, w_1+1j*gamma), (op_c, w_2+1j*gamma)], extra_terms=False
    #)
    #print(beta_tens)

    tpa_terms = (
        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - (w_f/2))
        + TransitionMoment(O, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - (w_f/2))
    )
    #tpa_tens = evaluate_property(scfres, "adc2", tpa_terms, [n], [], final_state=(f, 0))
    #print(tpa_tens)
    #excited_state = Excitation(state, 0, "adc2")
    #tpa_ref = tpa_resonant(excited_state)
    #print(tpa_ref)
    #np.testing.assert_allclose(tpa_tens, tpa_ref[1], atol=1e-7)

    esp_terms = (
        TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, f) / (w_n - w_f - w - 1j*gamma)
        + TransitionMoment(f, op_b, n) * TransitionMoment(n, op_a, f) / (w_n - w_f + w + 1j*gamma)
    )
    #esp_tens = evaluate_property(scfres, "adc2", esp_terms, [n], omega_alpha, gamma_val=0.124/Hartree, final_state=(f, 0))
    #print(esp_tens)
