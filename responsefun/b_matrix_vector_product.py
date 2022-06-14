import numpy as np
from itertools import product

import adcc
from adcc import AdcMethod
from adcc import LazyMp
from adcc import block as b
from adcc.functions import einsum
from adcc.AmplitudeVector import AmplitudeVector


def bmvp_adc0(ground_state, dip, vec):
    assert type(vec) == AmplitudeVector
    ph = (
            einsum('ac,ic->ia', dip.vv, vec.ph) 
            - 1.0 * einsum('ik,ka->ia', dip.oo, vec.ph)
    )
    return AmplitudeVector(ph=ph)


def bmvp_adc2(ground_state, dip, vec):
    assert type(vec) == AmplitudeVector
    if not dip.is_symmetric:
        raise NotImplementedError("b_matrix_vector_product is only implemented for symmetric one-particle operators.")
    p0 = ground_state.mp2_diffdm
    t2 = ground_state.t2(b.oovv)

    ph = (
            # product of the ph diagonal block with the singles block of the vector
            + 1.0 * einsum('ac,ic->ia', dip.vv, vec.ph)
            - 1.0 * einsum('ik,ka->ia', dip.oo, vec.ph)
            - 1.0 * einsum('ic,ja,jc->ia', vec.ph, p0.ov, dip.ov)
            - 1.0 * einsum('ic,jc,ja->ia', vec.ph, p0.ov, dip.ov)
            - 1.0 * einsum('ka,ib,kb->ia', vec.ph, p0.ov, dip.ov)
            - 1.0 * einsum('ka,kb,ib->ia', vec.ph, p0.ov, dip.ov)
            - 0.25 * einsum('ic,mnef,mnaf,ec->ia', vec.ph, t2, t2, dip.vv)
            - 0.25 * einsum('ic,mnef,mncf,ea->ia', vec.ph, t2, t2, dip.vv)
            - 0.5 * einsum('ic,mnce,mnaf,ef->ia', vec.ph, t2, t2, dip.vv)
            + 1.0 * einsum('ic,mncf,jnaf,jm->ia', vec.ph, t2, t2, dip.oo)
            + 0.25 * einsum('ka,mnef,inef,km->ia', vec.ph, t2, t2, dip.oo)
            + 0.25 * einsum('ka,mnef,knef,im->ia', vec.ph, t2, t2, dip.oo)
            - 1.0 * einsum('ka,knef,indf,ed->ia', vec.ph, t2, t2, dip.vv)
            + 0.5 * einsum('ka,knef,imef,mn->ia', vec.ph, t2, t2, dip.oo)
            + 0.5 * einsum('kc,knef,inaf,ec->ia', vec.ph, t2, t2, dip.vv)
            - 0.5 * einsum('kc,mncf,inaf,km->ia', vec.ph, t2, t2, dip.oo)
            + 0.5 * einsum('kc,inef,kncf,ea->ia', vec.ph, t2, t2, dip.vv)
            - 0.5 * einsum('kc,mnaf,kncf,im->ia', vec.ph, t2, t2, dip.oo)
            - 1.0 * einsum('kc,kncf,imaf,mn->ia', vec.ph, t2, t2, dip.oo)
            + 1.0 * einsum('kc,knce,inaf,ef->ia', vec.ph, t2, t2, dip.vv)
            
            # product of the ph-2p2h coupling block with the doubles block of the vector
            + 0.5 * (
                    - 2.0 * einsum('ilad,ld->ia', vec.pphh, dip.ov)
                    + 2.0 * einsum('ilad,lndf,nf->ia', vec.pphh, t2, dip.ov)
                    + 2.0 * einsum('ilca,lc->ia', vec.pphh, dip.ov)
                    - 2.0 * einsum('ilca,lncf,nf->ia', vec.pphh, t2, dip.ov)
                    - 2.0 * einsum('klad,kled,ie->ia', vec.pphh, t2, dip.ov)
                    - 2.0 * einsum('ilcd,nlcd,na->ia', vec.pphh, t2, dip.ov)
            )
    )

    pphh = (
            # product of the 2p2h-ph coupling block with the singles block of the vector
            + 0.5 * (
                (
                    - 1.0 * einsum('ia,jb->ijab', vec.ph, dip.ov)
                    + 1.0 * einsum('ia,jnbf,nf->ijab', vec.ph, t2, dip.ov)#
                    + 1.0 * einsum('ja,ib->ijab', vec.ph, dip.ov)
                    - 1.0 * einsum('ja,inbf,nf->ijab', vec.ph, t2, dip.ov)
                    + 1.0 * einsum('ib,ja->ijab', vec.ph, dip.ov)
                    - 1.0 * einsum('ib,jnaf,nf->ijab', vec.ph, t2, dip.ov)#
                    - 1.0 * einsum('jb,ia->ijab', vec.ph, dip.ov)
                    + 1.0 * einsum('jb,inaf,nf->ijab', vec.ph, t2, dip.ov)
                ).antisymmetrise(0,1).antisymmetrise(2,3)
                +(
                    - 1.0 * einsum('ka,ijeb,ke->ijab', vec.ph, t2, dip.ov)#
                    + 1.0 * einsum('kb,ijea,ke->ijab', vec.ph, t2, dip.ov)
                ).antisymmetrise(2,3)
                +(
                    - 1.0 * einsum('ic,njab,nc->ijab', vec.ph, t2, dip.ov)#
                    + 1.0 * einsum('jc,niab,nc->ijab', vec.ph, t2, dip.ov)
                ).antisymmetrise(0,1)
            )

            # product of the 2p2h diagonal block with the doubles block of the vector
            + 0.5 * (
                (
                    + 2.0 * einsum('ac,ijcb->ijab', dip.vv, vec.pphh)
                    - 2.0 * einsum('bc,ijca->ijab', dip.vv, vec.pphh)
                ).antisymmetrise(2,3)
                +(
                    - 2.0 * einsum('ki,kjab->ijab', dip.oo, vec.pphh)
                    + 2.0 * einsum('kj,kiab->ijab', dip.oo, vec.pphh)
                ).antisymmetrise(0,1)
            )
    )
    return AmplitudeVector(ph=ph, pphh=pphh)


DISPATCH = {
    "adc0": bmvp_adc0,
    "adc1": bmvp_adc0,
    "adc2": bmvp_adc2,
}

    
def b_matrix_vector_product(method, ground_state, dips, vecs):
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if method.name not in DISPATCH:
        raise NotImplementedError(f"b_matrix_vector_product is not implemented for {method.name}.")
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(dips, list):
        dips = [dips]
    if not isinstance(vecs, np.ndarray):
        vecs = np.array(vecs)

    comp_list_dips = list(range(len(dips)))
    comp_list_vecs = [list(range(shape)) for shape in vecs.shape]
    comp = list(product(comp_list_dips, *comp_list_vecs))

    ret_shape = (len(dips), *vecs.shape)
    ret = np.empty(ret_shape, dtype=object)
    
    for c in comp:
        dip = dips[c[0]]
        vec = vecs[c[1:]]
        ret[c] = DISPATCH[method.name](ground_state, dip, vec)

    return ret


if __name__ == "__main__":
    from pyscf import gto, scf
    import adcc
    from adcc.OneParticleOperator import product_trace
    from adcc.adc_pp import modified_transition_moments
    from respondo.solve_response import solve_response
    from responsefun.symbols_and_labels import *
    from responsefun.sum_over_states import TransitionMoment
    from responsefun.evaluate_property import evaluate_property_isr

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
    state = adcc.run_adc(scfres, method=method, n_singlets=5)
    matrix = adcc.AdcMatrix(method, refstate)
    mp = state.ground_state
    dips = state.reference_state.operators.electric_dipole
    magdips = state.reference_state.operators.magnetic_dipole
    mtms = modified_transition_moments(method, mp, dips)
    
    # test state difference dipole moments
    product_vecs = b_matrix_vector_product(method, mp, dips, state.excitation_vector)

    for excitation in state.excitations:
        dipmom = [
                excitation.excitation_vector @ pr
                for pr in product_vecs[:, excitation.index]
        ]
        diffdm = excitation.state_diffdm
        dipmom_ref = [
                product_trace(diffdm, dip) for dip in dips
        ]
        np.testing.assert_allclose(
            dipmom, dipmom_ref, atol=1e-12
        )

    # test the first term of the first-order hyperpolarizability
    omega_o = 1.0
    omega_2 = 0.5
    rvecs1 = [solve_response(matrix, rhs, omega_2, gamma=0.0) for rhs in mtms]
    components1 = list(product([0, 1, 2], repeat=2))
    product_bmatrix_rvecs1 = b_matrix_vector_product(method, mp, dips, rvecs1)
    beta_tens1 = np.zeros((3,3,3))
    for c in components1:
        rhs = product_bmatrix_rvecs1[c]
        rvec2 = solve_response(matrix, rhs, omega_o, gamma=0.0)
        for A in range(3):
            beta_tens1[A][c] = mtms[A] @ rvec2
    
    beta_sos_term = (
            TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o) * (w_k - w_2))
    )
    omegas_beta = [(w_o, omega_o), (w_2, omega_2)]
    beta_tens1_ref = evaluate_property_isr(state, beta_sos_term, [n, k], omegas_beta, extra_terms=False)
    np.testing.assert_allclose(beta_tens1, beta_tens1_ref, atol=1e-7)
