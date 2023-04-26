# taken from PR #158 of adcc as long as it has not been merged yet

import warnings
from math import sqrt

from adcc import block as b
from adcc.adc_pp.util import check_doubles_amplitudes, check_singles_amplitudes
from adcc.AdcMethod import AdcMethod
from adcc.AmplitudeVector import AmplitudeVector
from adcc.functions import einsum
from adcc.Intermediates import Intermediates
from adcc.LazyMp import LazyMp
from adcc.OneParticleOperator import OneParticleOperator


def tdm_adc0(mp, amplitude, intermediates):
    # C is either c(ore) or o(ccupied)
    C = b.c if mp.has_core_occupied_space else b.o
    check_singles_amplitudes([C, b.v], amplitude)
    u1 = amplitude.ph

    # Transition density matrix for (CVS-)ADC(0)
    dm = OneParticleOperator(mp, is_symmetric=False)
    dm[b.v + C] = u1.transpose()
    return dm


def tdm_adc1(mp, amplitude, intermediates):
    dm = tdm_adc0(mp, amplitude, intermediates)  # Get ADC(0) result
    # adc1_dp0_ov
    dm.ov = -einsum("ijab,jb->ia", mp.t2(b.oovv), amplitude.ph)
    return dm


def tdm_cvs_adc2(mp, amplitude, intermediates):
    # Get CVS-ADC(1) result (same as CVS-ADC(0))
    dm = tdm_adc0(mp, amplitude, intermediates)
    check_doubles_amplitudes([b.o, b.c, b.v, b.v], amplitude)
    u1 = amplitude.ph
    u2 = amplitude.pphh

    t2 = mp.t2(b.oovv)
    p0 = intermediates.cvs_p0

    # Compute CVS-ADC(2) tdm
    dm.oc = -einsum("ja,Ia->jI", p0.ov, u1) + (1 / sqrt(2)) * einsum(  # cvs_adc2_dp0_oc
        "kIab,jkab->jI", u2, t2
    )

    # cvs_adc2_dp0_vc
    dm.vc -= 0.5 * einsum("ab,Ib->aI", p0.vv, u1)
    return dm


def tdm_adc2(mp, amplitude, intermediates):
    dm = tdm_adc1(mp, amplitude, intermediates)  # Get ADC(1) result
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    u1 = amplitude.ph
    u2 = amplitude.pphh

    t2 = mp.t2(b.oovv)
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm

    # Compute ADC(2) tdm
    dm.oo = -einsum("ia,ja->ij", p0.ov, u1) - einsum("ikab,jkab->ji", u2, t2)  # adc2_dp0_oo
    dm.vv = +einsum("ia,ib->ab", u1, p0.ov) + einsum("ijac,ijbc->ab", u2, t2)  # adc2_dp0_vv
    dm.ov -= einsum("ijab,jb->ia", td2, u1)  # adc2_dp0_ov
    dm.vo += 0.5 * (  # adc2_dp0_vo
        +einsum("ijab,jkbc,kc->ai", t2, t2, u1)
        - einsum("ab,ib->ai", p0.vv, u1)
        + einsum("ja,ij->ai", u1, p0.oo)
    )
    return dm


DISPATCH = {
    "adc0": tdm_adc0,
    "adc1": tdm_adc1,
    "adc2": tdm_adc2,
    "adc2x": tdm_adc2,
    "cvs-adc0": tdm_adc0,
    "cvs-adc1": tdm_adc0,  # No extra contribs for CVS-ADC(1)
    "cvs-adc2": tdm_cvs_adc2,
    "cvs-adc2x": tdm_cvs_adc2,
}


def transition_dm(method, ground_state, amplitude, intermediates=None):
    """Compute the one-particle transition density matrix from ground to excited state in the MO
    basis.

    Parameters
    ----------
    method : str, AdcMethod
        The method to use for the computation (e.g. "adc2")
    ground_state : LazyMp
        The ground state upon which the excitation was based
    amplitude : AmplitudeVector
        The amplitude vector
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    warnings.warn("This function will soon be deprecated once PR #158 of adcc has been merged.")

    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude, AmplitudeVector):
        raise TypeError("amplitude should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError("transition_dm is not implemented " f"for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
