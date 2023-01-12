import numpy as np
#from adcc.adc_pp import modified_transition_moments
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.OneParticleOperator import product_trace
from responsefun.testdata.cache import MockExcitedStates
from responsefun.magnetic_dipole_moments import gs_magnetic_dipole_moment
from tqdm import tqdm
from cached_property import cached_property


# dict of operators available in responsefun so far
# the first argument specifies the symbol that is to be used for printing
# the second argument specifies the symmetry:
#   0: no symmetry assumed
#   1: hermitian
#   2: anti-hermitian
# the third argument specifies the dimensionality
available_operators = {
        "electric": ("\\mu", 1, 1),
        "magnetic": ("m", 2, 1),
        "dia_magnet": ("\\xi", 1, 2)
}


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


class AdccProperties:
    """
    Class encompassing all properties that can be obtained from adcc for a given operator.
    """
    def __init__(self, state, op_type):
        """
        Parameters
        ----------
        state: <class 'adcc.ExcitedStates.ExcitedStates'>
            ExcitedStates object returned by an ADC calculation.

        op_type: string
            String specifying the corresponding operator.
            It must be contained in the available_operators dict.
        """
        if op_type not in available_operators:
            raise NotImplementedError(
                    f"Only the following operators are available so far: {available_operators}."
            )
        self._state = state
        self._state_size = len(state.excitation_energy_uncorrected)

        self._op_type = op_type
        self._op_dim = available_operators[op_type][2]

        # to make things faster if not all state-to-state transition moments are needed
        # but only from or to a specific state
        self._s2s_tm_i = np.empty((self._state_size), dtype=object)
        self._s2s_tm_f = np.empty((self._state_size), dtype=object)

    @property
    def op_type(self):
        return self._op_type

    @property
    def op_dim(self):
        return self._op_dim

    @cached_property
    def operator(self):
        if self._op_type == "electric":
            return self._state.reference_state.operators.electric_dipole
        elif self._op_type == "magnetic":
            return self._state.reference_state.operators.magnetic_dipole
        else:
            raise NotImplementedError()

    @cached_property
    def gs_moment(self):
        if isinstance(self._state, MockExcitedStates):
            pm_level = self._state.property_method.replace("adc", "")
            if self._op_type == "electric":
                gs_moment = self._state.ground_state.dipole_moment[pm_level]
            elif self._op_type == "magnetic":
                gs_moment = gs_magnetic_dipole_moment(self._state.ground_state, pm_level)
            else:
                raise NotImplementedError()
        else:
            pm_level = self._state.property_method.level
            if self._op_type == "electric":
                gs_moment = self._state.ground_state.dipole_moment(pm_level)
            elif self._op_type == "magnetic":
                gs_moment = gs_magnetic_dipole_moment(self._state.ground_state, pm_level)
            else:
                raise NotImplementedError()
        return gs_moment 

    @cached_property
    def transition_moment(self):
        if self.op_type == "electric":
            transition_moment = self._state.transition_dipole_moment
        elif self.op_type == "magnetic":
            transition_moment = self._state.transition_magnetic_dipole_moment
        else:
            raise NotImplementedError()
        return transition_moment

    @cached_property
    def state_to_state_transition_moment(self):
        if isinstance(self._state, MockExcitedStates):
            if self.op_type == "electric":
                return self._state.transition_dipole_moment_s2s
            elif self.op_type == "magnetic":
                return self._state.transition_magnetic_moment_s2s
            else:
                raise NotImplementedError()
        return state_to_state_transition_moments(self._state, self.operator)

    def s2s_tm(self, initial_state=None, final_state=None):
        if initial_state is None and final_state is None:
            return self.state_to_state_transition_moment
        elif initial_state is None:
            if isinstance(self._state, MockExcitedStates):
                return self.state_to_state_transition_moment[:, final_state]
            if self._s2s_tm_f[final_state] is None:
                self._s2s_tm_f[final_state] = state_to_state_transition_moments(
                    self._state, self.operator, final_state=final_state
                )
            return self._s2s_tm_f[final_state]
        elif final_state is None:
            if isinstance(self._state, MockExcitedStates):
                return self.state_to_state_transition_moment[initial_state, :]
            if self._s2s_tm_i[initial_state] is None:
                self._s2s_tm_i[initial_state] = state_to_state_transition_moments(
                    self._state, self.operator, initial_state=initial_state
                )
            return self._s2s_tm_i[initial_state]
        else:
            if isinstance(self._state, MockExcitedStates):
                return self.state_to_state_transition_moment[initial_state, final_state]
            s2s_tm = state_to_state_transition_moments(
                self._state, self.operator, initial_state, final_state
            )
            return s2s_tm


if __name__ == "__main__":
    from pyscf import gto, scf
    import adcc
    from responsefun.testdata import cache
    from adcc.Excitation import Excitation

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
    mp = state.ground_state

    adcc_prop = AdccProperties(state, "electric")
    s2s_tdms = adcc_prop.state_to_state_transition_moment
    print(s2s_tdms)
