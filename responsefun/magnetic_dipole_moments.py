import numpy as np
from adcc.adc_pp.modified_transition_moments import modified_transition_moments
from adcc.functions import einsum
from adcc import AmplitudeVector
from adcc.LazyMp import LazyMp
from adcc.AdcMethod import AdcMethod
from adcc.OneParticleOperator import product_trace

# modified magnetic transition moments are taken from respondo

def compute_adc1_f1_mag(magdip, ground_state):
    mtm = magdip.ov + einsum(
        "ijab,jb->ia", ground_state.t2("o1o1v1v1"), magdip.ov
    )
    return AmplitudeVector(ph=mtm)


def compute_adc2_f1_mag(magdip, ground_state):
    t2 = ground_state.t2("o1o1v1v1")
    td2 = ground_state.td2("o1o1v1v1")
    p0 = ground_state.mp2_diffdm
    d = magdip
    return (
        d.ov
        + 1.0 * einsum("ijab,jb->ia", t2, d.ov + 0.5 * einsum("jkbc,kc->jb", t2, d.ov))
        + 0.5
        * (
            einsum("ij,ja->ia", p0.oo, d.ov)
            - 1.0 * einsum("ib,ab->ia", d.ov, p0.vv)
        )
        - 1.0 * einsum("ib,ab->ia", p0.ov, d.vv)
        - 1.0 * einsum("ij,ja->ia", d.oo, p0.ov)
        + 1.0 * einsum("ijab,jb->ia", td2, d.ov)
    )


def compute_adc2_f2_mag(magdip, ground_state):
    t2 = ground_state.t2("o1o1v1v1")
    term1 = -1.0 * einsum("ijac,bc->ijab", t2, magdip.vv)
    term2 = -1.0 * einsum("ik,kjab->ijab", magdip.oo, t2)
    term1 = term1.antisymmetrise(2, 3)
    term2 = term2.antisymmetrise(0, 1)
    return term1 - term2


def modified_magnetic_transition_moments(method, ground_state, dips_mag):
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(dips_mag, list):
        dips_mag = [dips_mag]

    if method.name == "adc0":
        mtms_mag = modified_transition_moments(
            method, ground_state, dips_mag
        )
        mtms_mag = [-1.0 * mtm_mag for mtm_mag in mtms_mag]
    elif method.name == "adc1":
        mtms_mag = [
            -1.0 * compute_adc1_f1_mag(mag, ground_state)
            for mag in dips_mag
        ]
    elif method.name == "adc2":
        mtms_mag = [
            -1.0
            * AmplitudeVector(
                ph=compute_adc2_f1_mag(mag, ground_state),
                pphh=compute_adc2_f2_mag(mag, ground_state),
            )
            for mag in dips_mag
        ]
    else:
        raise NotImplementedError("")

    return mtms_mag


#TODO: testing
def gs_magnetic_dipole_moment(ground_state, level=2):
    magdips = ground_state.reference_state.operators.magnetic_dipole
    ref_dipmom = np.array(
            [product_trace(dip, ground_state.reference_state.density) for dip in magdips]
    )
    if level == 1:
        return ref_dipmom
    elif level == 2:
        mp2corr = np.array(
                [product_trace(dip, ground_state.mp2_diffdm) for dip in magdips]
        )
        return ref_dipmom + mp2corr
    else:
        raise NotImplementedError("Only magnetic dipole moments for level 1 and 2"
                                  " are implemented.")
