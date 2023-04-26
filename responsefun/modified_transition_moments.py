# taken from PR #158 of adcc as long as it has not been merged yet

import warnings
from math import sqrt

from adcc import block as b
from adcc.AdcMethod import AdcMethod
from adcc.AmplitudeVector import AmplitudeVector
from adcc.functions import einsum, evaluate
from adcc.Intermediates import Intermediates
from adcc.LazyMp import LazyMp


def mtm_adc0(mp, op, intermediates):
    return (
        AmplitudeVector(ph=op.ov)
        if op.is_symmetric
        else AmplitudeVector(ph=op.vo.transpose((1, 0)))
    )


def mtm_adc1(mp, op, intermediates):
    if op.is_symmetric:
        f1 = op.ov
    else:
        f1 = op.vo.transpose((1, 0))
    f1 -= einsum("ijab,jb->ia", mp.t2(b.oovv), op.ov)
    return AmplitudeVector(ph=f1)


def mtm_adc2(mp, op, intermediates):
    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm

    if op.is_symmetric:
        op_vo = op.ov.transpose((1, 0))
    else:
        op_vo = op.vo

    f1 = (
        +op_vo.transpose((1, 0))
        - einsum("ijab,jb->ia", t2, +op.ov - 0.5 * einsum("jkbc,ck->jb", t2, op_vo))
        + 0.5 * einsum("ij,aj->ia", p0.oo, op_vo)
        - 0.5 * einsum("bi,ab->ia", op_vo, p0.vv)
        + einsum("ib,ab->ia", p0.ov, op.vv)
        - einsum("ji,ja->ia", op.oo, p0.ov)
        - einsum("ijab,jb->ia", mp.td2(b.oovv), op.ov)
    )
    f2 = +einsum("ijac,bc->ijab", t2, op.vv).antisymmetrise(2, 3) + einsum(
        "ki,jkab->ijab", op.oo, t2
    ).antisymmetrise(0, 1)
    return AmplitudeVector(ph=f1, pphh=f2)


def mtm_cvs_adc0(mp, op, intermediates):
    return AmplitudeVector(ph=op.cv)


def mtm_cvs_adc2(mp, op, intermediates):
    f1 = (
        +op.cv
        - einsum("Ib,ba->Ia", op.cv, intermediates.cvs_p0.vv)
        - einsum("Ij,ja->Ia", op.co, intermediates.cvs_p0.ov)
    )
    f2 = (1 / sqrt(2)) * einsum("Ik,kjab->jIab", op.co, mp.t2(b.oovv))
    return AmplitudeVector(ph=f1, pphh=f2)


DISPATCH = {
    "adc0": mtm_adc0,
    "adc1": mtm_adc1,
    "adc2": mtm_adc2,
    "adc2x": mtm_adc2,
    "cvs-adc0": mtm_cvs_adc0,
    "cvs-adc1": mtm_cvs_adc0,  # Identical to CVS-ADC(0)
    "cvs-adc2": mtm_cvs_adc2,
    "cvs-adc2x": mtm_cvs_adc2,
}


def modified_transition_moments(method, ground_state, operator=None, intermediates=None):
    """Compute the modified transition moments (MTM) for the provided ADC method with reference to
    the passed ground state.

    Parameters
    ----------
    method: adc.Method
        Provide a method at which to compute the MTMs
    ground_state : adcc.LazyMp
        The MP ground state
    operator : adcc.OneParticleOperator or list, optional
        Only required if different operators than the standard
        electric dipole operators in the MO basis should be used.
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse

    Returns
    -------
    adcc.AmplitudeVector or list of adcc.AmplitudeVector
    """
    warnings.warn("This function will soon be deprecated once PR #158 of adcc has been merged.")

    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    unpack = False
    if operator is None:
        operator = ground_state.reference_state.operators.electric_dipole
    elif not isinstance(operator, list):
        unpack = True
        operator = [operator]
    if method.name not in DISPATCH:
        raise NotImplementedError(
            "modified_transition_moments is not " f"implemented for {method.name}."
        )

    ret = [DISPATCH[method.name](ground_state, op, intermediates) for op in operator]
    if unpack:
        assert len(ret) == 1
        ret = ret[0]
    return evaluate(ret)
