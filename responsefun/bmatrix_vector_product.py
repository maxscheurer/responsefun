import numpy as np
import adcc
from adcc import AdcMethod
from adcc import LazyMp
from adcc import block as b
from adcc.functions import einsum
from adcc.AmplitudeVector import AmplitudeVector
from respondo.cpp_algebra import ResponseVector as RV


def bmvp_adc0(ground_state, dip, vec):
    assert isinstance(vec, AmplitudeVector)
    ph = (
            + 1.0 * einsum('ic,ac->ia', vec.ph, dip.vv) 
            - 1.0 * einsum('ka,ki->ia', vec.ph, dip.oo)
    )
    return AmplitudeVector(ph=ph)


def bmvp_adc2(ground_state, dip, vec):
    assert isinstance(vec, AmplitudeVector)
    if dip.is_symmetric:
        dip_vo = dip.ov.transpose((1, 0))
    else:
        dip_vo = dip.vo.copy()
    p0 = ground_state.mp2_diffdm
    t2 = ground_state.t2(b.oovv)

    ph = (
            # product of the ph diagonal block with the singles block of the vector
            # zeroth order
            + 1.0 * einsum('ic,ac->ia', vec.ph, dip.vv)
            - 1.0 * einsum('ka,ki->ia', vec.ph, dip.oo)
            # second order
            # (2,1)
            - 1.0 * einsum('ic,jc,aj->ia', vec.ph, p0.ov, dip_vo)
            - 1.0 * einsum('ka,kb,bi->ia', vec.ph, p0.ov, dip_vo)
            - 1.0 * einsum('ic,ja,jc->ia', vec.ph, p0.ov, dip.ov) # h.c.
            - 1.0 * einsum('ka,ib,kb->ia', vec.ph, p0.ov, dip.ov) # h.c.
            # (2,2)
            - 0.25 * einsum('ic,mnef,mnaf,ec->ia', vec.ph, t2, t2, dip.vv)
            - 0.25 * einsum('ic,mnef,mncf,ae->ia', vec.ph, t2, t2, dip.vv) # h.c.
            # (2,3)
            - 0.5 * einsum('ic,mnce,mnaf,ef->ia', vec.ph, t2, t2, dip.vv)
            + 1.0 * einsum('ic,mncf,jnaf,jm->ia', vec.ph, t2, t2, dip.oo)
            # (2,4)
            + 0.25 * einsum('ka,mnef,inef,km->ia', vec.ph, t2, t2, dip.oo)
            + 0.25 * einsum('ka,mnef,knef,mi->ia', vec.ph, t2, t2, dip.oo) # h.c.
            # (2,5)
            - 1.0 * einsum('ka,knef,indf,ed->ia', vec.ph, t2, t2, dip.vv)
            + 0.5 * einsum('ka,knef,imef,mn->ia', vec.ph, t2, t2, dip.oo)
            # (2,6)
            + 0.5 * einsum('kc,knef,inaf,ec->ia', vec.ph, t2, t2, dip.vv)
            - 0.5 * einsum('kc,mncf,inaf,km->ia', vec.ph, t2, t2, dip.oo)
            + 0.5 * einsum('kc,inef,kncf,ae->ia', vec.ph, t2, t2, dip.vv) # h.c.
            - 0.5 * einsum('kc,mnaf,kncf,mi->ia', vec.ph, t2, t2, dip.oo) # h.c.
            # (2,7)
            - 1.0 * einsum('kc,kncf,imaf,mn->ia', vec.ph, t2, t2, dip.oo)
            + 1.0 * einsum('kc,knce,inaf,ef->ia', vec.ph, t2, t2, dip.vv)

            # product of the ph-2p2h coupling block with the doubles block of the vector
            + 0.5 * (
                - 2.0 * einsum('ilad,ld->ia', vec.pphh, dip.ov)
                + 2.0 * einsum('ilad,lndf,fn->ia', vec.pphh, t2, dip_vo)
                + 2.0 * einsum('ilca,lc->ia', vec.pphh, dip.ov)
                - 2.0 * einsum('ilca,lncf,fn->ia', vec.pphh, t2, dip_vo)
                - 2.0 * einsum('klad,kled,ei->ia', vec.pphh, t2, dip_vo)
                - 2.0 * einsum('ilcd,nlcd,an->ia', vec.pphh, t2, dip_vo)
            )
    )

    pphh = (
            # product of the 2p2h-ph coupling block with the singles block of the vector
            + 0.5 * (
                (
                    - 1.0 * einsum('ia,bj->ijab', vec.ph, dip_vo)
                    + 1.0 * einsum('ia,jnbf,nf->ijab', vec.ph, t2, dip.ov)
                    + 1.0 * einsum('ja,bi->ijab', vec.ph, dip_vo)
                    - 1.0 * einsum('ja,inbf,nf->ijab', vec.ph, t2, dip.ov)
                    + 1.0 * einsum('ib,aj->ijab', vec.ph, dip_vo)
                    - 1.0 * einsum('ib,jnaf,nf->ijab', vec.ph, t2, dip.ov)
                    - 1.0 * einsum('jb,ai->ijab', vec.ph, dip_vo)
                    + 1.0 * einsum('jb,inaf,nf->ijab', vec.ph, t2, dip.ov)
                ).antisymmetrise(0,1).antisymmetrise(2,3)
                +(
                    - 1.0 * einsum('ka,ijeb,ke->ijab', vec.ph, t2, dip.ov)
                    + 1.0 * einsum('kb,ijea,ke->ijab', vec.ph, t2, dip.ov)
                ).antisymmetrise(2,3)
                +(
                    - 1.0 * einsum('ic,njab,nc->ijab', vec.ph, t2, dip.ov)
                    + 1.0 * einsum('jc,niab,nc->ijab', vec.ph, t2, dip.ov)
                ).antisymmetrise(0,1)
            )

            # product of the 2p2h diagonal block with the doubles block of the vector
            + 0.5 * (
                (
                    + 2.0 * einsum('ijcb,ac->ijab', vec.pphh, dip.vv)
                    - 2.0 * einsum('ijca,bc->ijab', vec.pphh, dip.vv)
                ).antisymmetrise(2,3)
                +(
                    - 2.0 * einsum('kjab,ki->ijab', vec.pphh, dip.oo)
                    + 2.0 * einsum('kiab,kj->ijab', vec.pphh, dip.oo)
                ).antisymmetrise(0,1)
            )
    )
    return AmplitudeVector(ph=ph, pphh=pphh)


DISPATCH = {
    "adc0": bmvp_adc0,
    "adc1": bmvp_adc0, # identical to ADC(0)
    "adc2": bmvp_adc2,
}

    
def bmatrix_vector_product(method, ground_state, dips, vec):
    """Compute the matrix-vector product of an ISR one-particle operator
    for the provided ADC method.
    The product was derived using the original equations from the work of
    Schirmer and Trofimov (J. Schirmer and A. B. Trofimov, “Intermediate state
    representation approach to physical properties of electronically excited
    molecules,” J. Chem. Phys. 120, 11449–11464 (2004).).

    Parameters
    ----------
    method: str, AdcMethod
        The  method to use for the computation of the matrix-vector product
    ground_state : adcc.LazyMp
        The MP ground state
    dips : OneParticleOperator or list of OneParticleOperator
        One-particle matrix elements associated with the dipole operator        
    vec: AmplitudeVector
        A vector with singles and doubles block
    Returns
    -------
    adcc.AmplitudeVector or list of adcc.AmplitudeVector
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if method.name not in DISPATCH:
        raise NotImplementedError(f"b_matrix_vector_product is not implemented for {method.name}.")
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    unpack = False
    if not isinstance(dips, list):
        unpack = True
        dips = [dips]

    ret = [DISPATCH[method.name](ground_state, dip, vec) for dip in dips]
    if unpack:
        assert len(ret) == 1
        ret = ret[0]
    return ret


#TODO: testing (however, since solve_response can only handle real right-hand sides, it is difficult) 
def bmatrix_vector_product_complex(method, ground_state, dips, vec):
    unpack = False
    if not isinstance(dips, list):
        unpack = True
        dips = [dips]
    assert isinstance(vec, RV)
    ret = []
    
    for dip in dips:
        product_real = bmatrix_vector_product(method, ground_state, dip, vec.real)
        product_imag = bmatrix_vector_product(method, ground_state, dip, vec.imag)
        ret.append(RV(product_real, product_imag))
    if unpack:
        assert len(1) == 1
        ret = ret[0]
    return ret


if __name__ == "__main__":
    from pyscf import gto, scf
    import adcc
    from adcc.OneParticleOperator import product_trace
    from adcc.adc_pp import modified_transition_moments
    from respondo.solve_response import solve_response, transition_polarizability
    from respondo.misc import select_property_method
    from responsefun.symbols_and_labels import *
    from responsefun.sum_over_states import TransitionMoment
    from responsefun.evaluate_property import evaluate_property_isr
    from itertools import product

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
    method = "adc2"
    state = adcc.run_adc(scfres, method=method, n_singlets=10)
    matrix = adcc.AdcMatrix(method, refstate)
    mp = state.ground_state
    property_method = select_property_method(matrix)
    dips = state.reference_state.operators.electric_dipole
    magdips = state.reference_state.operators.magnetic_dipole
    mtms = modified_transition_moments(method, mp, dips)

    # test state difference dipole moments
    # electric
    for excitation in state.excitations:
        product_vecs = bmatrix_vector_product(method, mp, dips, excitation.excitation_vector)
        dipmom = [
                excitation.excitation_vector @ pr
                for pr in product_vecs
        ]
        diffdm = excitation.state_diffdm
        dipmom_ref = [
                product_trace(diffdm, dip) for dip in dips
        ]
        np.testing.assert_allclose(
            dipmom, dipmom_ref, atol=1e-12
        )

    # magnetic
    for excitation in state.excitations:
        product_vecs_mag = bmatrix_vector_product(method, mp, magdips, excitation.excitation_vector)
        dipmom = [
                excitation.excitation_vector @ pr
                for pr in product_vecs_mag
        ]
        diffdm = excitation.state_diffdm
        dipmom_ref = [
                product_trace(diffdm, dip) for dip in magdips
        ]
        np.testing.assert_allclose(
           dipmom, dipmom_ref, atol=1e-12
        )


    ## test the first term of the first-order hyperpolarizability
    #omega_o = 1.0
    #omega_2 = 0.5
    #rvecs1 = [solve_response(matrix, rhs, omega_2, gamma=0.0) for rhs in mtms]
    #components1 = list(product([0, 1, 2], repeat=2))
    #beta_tens1 = np.zeros((3,3,3))
    #for c in components1:
    #    rhs = bmatrix_vector_product(method, mp, dips[c[0]], rvecs1[c[1]])
    #    rvec2 = solve_response(matrix, rhs, omega_o, gamma=0.0)
    #    for A in range(3):
    #        beta_tens1[A][c] = mtms[A] @ rvec2
    #
    #beta_sos_term = (
    #        TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o) * (w_k - w_2))
    #)
    #omegas_beta = [(w_o, omega_o), (w_2, omega_2)]
    #beta_tens1_ref = evaluate_property_isr(state, beta_sos_term, [n, k], omegas_beta, extra_terms=False)
    #np.testing.assert_allclose(beta_tens1, beta_tens1_ref, atol=1e-7)


    ## test second term of the MCD B term
    #v_f = state.excitation_vector[2]
    #e_f = state.excitation_energy[2]
    #def projection(X, bl=None):
    #    if bl:
    #        vb = getattr(v_f, bl)
    #        return vb * (vb.dot(X)) / (vb.dot(vb))
    #    else:
    #        return v_f * (v_f @ X) / (v_f @ v_f)

    #response_el = [solve_response(matrix, rhs, e_f, gamma=0.0, projection=projection) for rhs in mtms]
    #product_mag = bmatrix_vector_product(method, mp, magdips, v_f)
    #mcd_bterm2 = np.zeros((3,3))
    #for A in range(3):
    #    for B in range(3):
    #        mcd_bterm2[A][B] = - (response_el[B] @ product_mag[A])
    #mcd_bterm2_ref = -transition_polarizability(property_method, mp, v_f, magdips, response_el)
    #print(mcd_bterm2)
    #print(mcd_bterm2_ref)
    #np.testing.assert_allclose(mcd_bterm2, mcd_bterm2_ref, atol=1e-12)


    # test b_matrix_vector_product_complex
    #rvecs_test = [solve_response(matrix, RV(rhs), omega_2, gamma=0.01) for rhs in mtms]
    #product_bmatrix_rvecs_test = [bmatrix_vector_product_complex(method, mp, dips, rvec) for rvec in rvecs_test]
    #print(product_bmatrix_rvecs_test)
