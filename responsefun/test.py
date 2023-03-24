import numpy as np
from itertools import product
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.OneParticleOperator import product_trace
from responsefun.testdata.cache import MockExcitedStates
from tqdm import tqdm
from cached_property import cached_property

from pyscf import gto, scf
import adcc

def state_to_state_transition_moments(state, operator, initial_state=None, final_state=None):
    istates = state.size
    excitations1 = state.excitations
    if initial_state is not None:
        istates = 1
        excitations1 = [state.excitations[initial_state]]
    fstates = state.size
    excitations2 = state.excitations
    if final_state is not None:
        fstates = 1
        excitations2 = [state.excitations[final_state]]

    s2s_tm = np.zeros((istates, fstates, 3))
    for i, ee1 in enumerate(tqdm(excitations1)):
        for j, ee2 in enumerate(excitations2):
            tdm = state2state_transition_dm(
                state.property_method,
                state.ground_state,
                ee1.excitation_vector,
                ee2.excitation_vector,
                state.matrix.intermediates,
            )
            s2s_tm[i, j] = np.array([product_trace(tdm, op) for op in operator])
    return np.squeeze(s2s_tm)

def state_to_state_transition_moments2(state, op_type, initial_state=None, final_state=None):
    if isinstance(state, MockExcitedStates):
        if op_type == "electric":
            s2s_tdms = state.transition_dipole_moment_s2s
        elif op_type == "magnetic":
            s2s_tdms = state.transition_magnetic_moment_s2s
        else:
            raise NotImplementedError()
        if initial_state is None and final_state is None:
            return s2s_tdms
        elif initial_state is None:
            return s2s_tdms[:, final_state]
        elif final_state is None:
            return s2s_tdms[initial_state, :]
        else:
            return s2s_tdms[initial_state, final_state]
    else:
        if op_type == "electric":
            dips = state.reference_state.operators.electric_dipole
        elif op_type == "magnetic":
            dips = state.reference_state.operators.magnetic_dipole
        else:
            raise NotImplementedError()
        if initial_state is None and final_state is None:
            s2s_tdms = np.zeros((state.size, state.size, 3))
            excitations1 = state.excitations
            excitations2 = state.excitations
        elif initial_state is None:
            s2s_tdms = np.zeros((state.size, 1, 3))
            excitations1 = state.excitations
            excitations2 = [state.excitations[final_state]]
        elif final_state is None:
            s2s_tdms = np.zeros((1, state.size, 3))
            excitations1 = [state.excitations[initial_state]]
            excitations2 = state.excitations
        else:
            s2s_tdms = np.zeros((1, 1, 3))
            excitations1 = [state.excitations[initial_state]]
            excitations2 = [state.excitations[final_state]]
        for i, ee1 in enumerate(tqdm(excitations1)):
            for j, ee2 in enumerate(excitations2):
                tdm = state2state_transition_dm(
                    state.property_method,
                    state.ground_state,
                    ee1.excitation_vector,
                    ee2.excitation_vector,
                    state.matrix.intermediates,
                )
                s2s_tdms[i, j] = np.array([product_trace(tdm, dip) for dip in dips])
        return np.squeeze(s2s_tdms)

def state_to_state_transition_moments3(state, operator, initial_state=None, final_state=None):
    istates = state.size
    excitations1 = state.excitations
    if initial_state is not None:
        istates = 1
        excitations1 = [state.excitations[initial_state]]
    fstates = state.size
    excitations2 = state.excitations
    if final_state is not None:
        fstates = 1
        excitations2 = [state.excitations[final_state]]

    op_shape = np.shape(operator)
    iterables = [list(range(shape)) for shape in op_shape]
    components = list(product(*iterables))
    s2s_tm = np.zeros((istates, fstates, *op_shape))
    for i, ee1 in enumerate(tqdm(excitations1)):
        for j, ee2 in enumerate(excitations2):
            tdm = state2state_transition_dm(
                state.property_method,
                state.ground_state,
                ee1.excitation_vector,
                ee2.excitation_vector,
                state.matrix.intermediates,
            )
            temp = np.zeros(op_shape)
            for c in components:
                c = c[0] if len(c) == 1 else c
                temp[c] = product_trace(tdm, operator[c])
            s2s_tm[i, j] = temp
    print(s2s_tm)
    return np.squeeze(s2s_tm)

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

s2s = state_to_state_transition_moments3(state, refstate.operators.magnetic_dipole)
s2s_2 = state_to_state_transition_moments(state, refstate.operators.magnetic_dipole)

print(s2s)
print(type(s2s[0]), type(s2s[0][0]), type(s2s[0][0][0]))
print(s2s_2)
print(type(s2s_2[0]), type(s2s_2[0][0]), type(s2s_2[0][0][0]))
np.testing.assert_allclose(s2s, s2s_2, atol=1e-8)