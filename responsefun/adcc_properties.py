import numpy as np
from adcc.adc_pp import modified_transition_moments
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.adc_pp.transition_dm import transition_dm
from adcc.OneParticleOperator import product_trace
from responsefun.testdata.cache import MockExcitedStates
from responsefun.magnetic_dipole_moments import modified_magnetic_transition_moments, gs_magnetic_dipole_moment
from tqdm import tqdm
from itertools import product

# dict of operators available in responsefun so far
# the first argument specifies the symbol that is to be used for printing
# the second argument specifies the symmetry:
#   0: no symmetry assumed
#   1: hermitian
#   2: anti-hermitian
# the third argument specifies the dimensionality
available_operators = {
        "r_r": ("r_dot_r", 1, 1),
        "r_quadr": ("r^2", 1, 2)
        "electric": ("\\mu", 1, 1),
        "magnetic": ("m", 2, 1),
        "dia_magnet": ("\\xi", 1, 2)
}


# TODO: make modified_transition_moments function also applicable for asymmetric operators
dispatch_mtms = {
        "electric": modified_transition_moments,
        "magnetic": modified_magnetic_transition_moments,
        "diag_mag": modified_transition_moments,
        "r_r": modified_transition_moments,
        "r_quadr": modified_transition_moments
}

def ground_state_moments(state, op_type):
    assert op_type in available_operators
    if op_type == "diag_magnet":
        op_int = state.reference_state.operators.diag_mag
        size = op_int.shape[0]
    elif op_type == "r_r":
        op_int = state.reference_state.operators.r_r
        size = op_int.shape[0]
    else:
        op_int = state.reference_state.operators.r_quadr
        size = op_int.size
    pm_level = state.property_method.level
    ref_state_density = state.reference_state.density
    components = list(product(range(size), repeat = op_int.ndim))
    for c in components:
        if op_int.ndim == 1:
            ref_state_moment = np.zeros((size))
            ref_state_moment[c[0]] =  -np.array(product_trace(op_int[c[0]], ref_state_density))
        else:
            ref_state_moment = np.zeros((size, size))
            ref_state_moment[c[0]][c[1]] = - np.array(product_trace(op_int[c[0]][c[1]], ref_state_density))
    if pm_level ==1:
        return  ref_state_moment
    if pm_level ==2:
        mp2_density = state.ground_state.density(2)
        for c in components:
            if op_int.ndim == 1:
                mp2_corr = np.zeros((size))
                mp2_corr[c[0]] = - np.array(product_trace(op_int[c[0]], mp2_density))
            else:
                mp2_corr = np.zeros((size, size))
                mp2_corr[c[0]][c[1]] = - np.array(product_trace(op_int[c[0]][c[1]], mp2_density))
        return ref_state_moment + mp2_corr
    else:
        raise NotImplementedError("Only dipole moments for level 1 and 2"
                                      " are implemented.")

def transition_moments(state, op_type):
    assert op_type in available_operators
    if isinstance(state, MockExcitedStates):
        if op_type == "diag_magnet":
            tdms = state.transition_moments_diag_mag
        elif op_type == "r_r":
            tdms = state.transition_moments_r_r
        else:
            tdms = state.transition_moment_r_quadr
    else:
        if op_type == "diag_magn":
            op_int= state.reference_state.operators.diag_mag
            size = op_int.shape[0]
        elif op_type == "r_r":
            
            op_int = state.reference_state.operators.r_r
            size = op_int.shape[0]
        else:
            op_int = state.reference_state.operators.r_quadr
            size = op_int.size
        if op_int.ndim == 2:
            tdms = np.zeros((state.size, size, size)
                    )
        else:
            tdms = np.zeros((state.size, size)
                    )
        components = list(product(range(size), repeat = op_int.ndim))
        print(components)
        for ee in tqdm(state.excitations):
            i = ee.index
            tdm = transition_dm(
                state.property_method,
                state.ground_state,
                ee.excitation_vector,
                state.matrix.intermediates,
                )
            for c in components:
                if op_int.ndim == 2:
                    tdms[i][c[0]][c[1]] = np.array(product_trace(tdm, op_int[c[0]][c[1]]))
                else:
                    tdms[i][c[0]] = np.array(product_trace(tdm, op_int[c[0]]))
        return np.squeeze(tdms)

def state_to_state_transition_moments(state, op_type, initial_state=None, final_state=None):
    assert op_type in available_operators
    if isinstance(state, MockExcitedStates):
        if op_type == "electric":
            s2s_tdms = state.transition_dipole_moment_s2s
        elif op_type == "magnetic":
            s2s_tdms = state.transition_magnetic_moment_s2s
        elif op_type == "electric":
            s2s_tdms = state.transition_dipole_moment_s2s
        elif op_type == "diag_magnet":
            s2s_tdms = state.diag_mag_s2s
        elif op_type == "r_r":
            s2s_tdms = state.r_r_s2s
        elif op_type == "r_quadr":
            s2s_tdms = state.r_quadr_s2s
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
        if op_type == "magnetic":
            op_int = np.asarray(state.reference_state.operators.magnetic_dipole)
            size = op_int.size
        elif op_type == "electric":
            op_int = np.asarray(state.reference_state.operators.electric_dipole)
            size = op_int.size
        elif op_type == "diag_magnet":
            op_int= state.reference_state.operators.diag_mag
            size = op_int.shape[0]
        elif op_type == "r_r":
            op_int = state.reference_state.operators.r_r
            size = op_int.shape[0]
        elif op_type == 'r_quadr':
            op_int = state.reference_state.operators.r_quadr
            size = op_int.size
        else:
            raise NotImplementedError()
        if initial_state is None and final_state is None:
            if op_int.ndim ==2:
                s2s_tdms = np.zeros((state.size, state.size, size, size))
            else:
                s2s_tdms = np.zeros((state.size, state.size, size))
            excitations1 = state.excitations
            excitations2 = state.excitations
        elif initial_state is None:
            if op_int.ndim ==2:
                s2s_tdms = np.zeros((state.size, 1, size, size))
            else:
                s2s_tdms = np.zeros((state.size, 1, size))
            excitations1 = state.excitations
            excitations2 = [state.excitations[final_state]]
        elif final_state is None:
            if op_int.ndim ==2:
                s2s_tdms = np.zeros((1, state.size, size, size))
            else:
                s2s_tdms = np.zeros((1, state.size, size))
            excitations1 = [state.excitations[initial_state]]
            excitations2 = state.excitations
        else:
            if op_int.ndim ==2:
                s2s_tdms = np.zeros((1, 1, size, size))
            else:
                s2s_tdms = np.zeros((1, 1, size))
            excitations1 = [state.excitations[initial_state]]
            excitations2 = [state.excitations[final_state]]
        components = list(product(range(size), repeat = op_int.ndim))
        for i, ee1 in enumerate(tqdm(excitations1)):
            for j, ee2 in enumerate(excitations2):
                tdm = state2state_transition_dm(
                    state.property_method,
                    state.ground_state,
                    ee1.excitation_vector,
                    ee2.excitation_vector,
                    state.matrix.intermediates,
                )
                for c in components:
                    if op_int.ndim ==2:
                        s2s_tdms[i][j][c[0]][c[1]] = np.array(product_trace(tdm, op_int[c[0]][c[1]]))
                    else:
                        s2s_tdms[i][j][c[0]] = np.array(product_trace(tdm, op_int[c[0]]))
        return np.squeeze(s2s_tdms)

class AdccProperties:
    """
    Class encompassing all properties that can be obtained from adcc for a given operator.
    When instantiated, the object is "empty" because the corresponding properties are not computed
    until they are needed for the first time.
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
        self._op_type = op_type
        self._op_dim = available_operators[op_type][2]
        
        self._dips = None
        self._mtms = None
        self._gs_dip_moment = None
        self._transition_dipole_moment = None
        self._state_size = len(state.excitation_energy_uncorrected)
        self._state_to_state_transition_moment = None
        # to make things faster if not all state-to-state transition moments are needed,
        # but only from or to a specific state
        self._s2s_tm_i = np.empty((self._state_size), dtype=object)
        self._s2s_tm_f = np.empty((self._state_size), dtype=object)

    @property
    def op_type(self):
        return self._op_type

    @property
    def op_dim(self):
        return self._op_dim

    @property
    def dips(self):
        if self._dips is None:
            if self._op_type == "electric":
                self._dips = self._state.reference_state.operators.electric_dipole
            elif self._op_type == "magnetic":
                self._dips = self._state.reference_state.operators.magnetic_dipole
            elif self._op_type == "electric":
                self._dips = self._state.reference_state.operators.electric_dipole
            elif self._op_type == "diag_magnet":
                self._dips = self._state.reference_state.operators.diag_mag
            elif self._op_type == "r_r":
                self._dips = self._state.reference_state.operators.r_r
            elif self._op_type == 'r_quadr':
                self._dips = self._state.reference_state.operators.r_quadr
            else:
                raise NotImplementedError()
        return self._dips

    @property
    def mtms(self):
        if self._mtms is None:
            self._mtms = dispatch_mtms[self._op_type](
                    self._state.property_method, self._state.ground_state, self.dips
            )
        return self._mtms

    @property
    def gs_dip_moment(self):
        if self._gs_dip_moment is None:
            if isinstance(self._state, MockExcitedStates):
                pm_level = self._state.property_method.replace("adc", "")
                if self._op_type == "electric":
                    self._gs_dip_moment = self._state.ground_state.dipole_moment[pm_level]
                elif self._op_type == "magnetic":
                    self._gs_dip_moment = gs_magnetic_dipole_moment(self._state.ground_state, pm_level)
                else:
                    raise NotImplementedError()
            else:
                pm_level = self._state.property_method.level
                if self._op_type == "electric":
                    self._gs_dip_moment = self._state.ground_state.dipole_moment(pm_level)
                elif self._op_type == "magnetic":
                    self._gs_dip_moment = gs_magnetic_dipole_moment(self._state.ground_state, pm_level)
                elif self._op_type == "diag_magnet" or self._op_type == "r_r" or self._op_type == 'r_quadr':
                    self._gs_dip_moment = ground_state_moments(self._state, self._op_type)
                else:
                    raise NotImplementedError()
        return self._gs_dip_moment 

    @property
    def transition_dipole_moment(self):
        if self._transition_dipole_moment is None:
            if self._op_type == "electric":
                self._transition_dipole_moment = self._state.transition_dipole_moment
            elif self._op_type == "magnetic":
                self._transition_dipole_moment = self._state.transition_magnetic_dipole_moment
            elif self._op_type == "diag_magnet" or self._op_type == "r_r" or self._op_type == 'r_quadr':
                self._transition_dipole_moment = transition_moments(self._state, self._op_type)
            else:
                raise NotImplementedError()
        return self._transition_dipole_moment

    @property
    def state_to_state_transition_moment(self):
        if self._state_to_state_transition_moment is None:
            self._state_to_state_transition_moment = (
                    state_to_state_transition_moments(self._state, self._op_type)
            )
        return self._state_to_state_transition_moment

    def s2s_tm(self, initial_state=None, final_state=None):
        if initial_state is None and final_state is None:
            return self.state_to_state_transition_moment
        elif initial_state is None:
            if self._s2s_tm_f[final_state] is None:
                if self._state_to_state_transition_moment is None:
                    self._s2s_tm_f[final_state] = (
                            state_to_state_transition_moments(self._state, self._op_type, final_state=final_state)
                    )
                else:
                    self._s2s_tm_f[final_state] = self._state_to_state_transition_moment[:, final_state]
            return self._s2s_tm_f[final_state]
        elif final_state is None:
            if self._s2s_tm_i[initial_state] is None:
                if self._state_to_state_transition_moment is None:
                    self._s2s_tm_i[initial_state] = (
                            state_to_state_transition_moments(self._state, self._op_type, initial_state=initial_state)
                    )
                else:
                    self._s2s_tm_i[initial_state] =  self._state_to_state_transition_moment[initial_state, :]
            return self._s2s_tm_i[initial_state]
        else:
            return state_to_state_transition_moments(self._state, self._op_type, initial_state, final_state)


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
    
    adcc_prop = AdccProperties(state, "r_quadr")
    tdms = adcc_prop.gs_dip_moment
    print(tdms)
    #s2s_tdms = adcc_prop.state_to_state_transition_moment
    #print(s2s_tdms)
