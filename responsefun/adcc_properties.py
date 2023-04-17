import numpy as np
from adcc.adc_pp import modified_transition_moments
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.adc_pp.transition_dm import transition_dm
from adcc.OneParticleOperator import product_trace
from responsefun.testdata.cache import MockExcitedStatesTestData
from responsefun.magnetic_dipole_moments import gs_magnetic_dipole_moment
from tqdm import tqdm
from itertools import product
from responsefun.MockExcitedStates import MockExcitedStates

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
        "dia_magnet": ("\\xi", 1, 2),
        "electric_quadrupole": ("Q", 1, 2),
        "electric_quadrupole_traceless": ("\\theta", 1, 2),
        "nabla": ("\\nabla", 2, 1)
}

def ground_state_moments(state, op_type):
    assert op_type in available_operators
    charges = np.array(state.reference_state.nuclear_charges)
    masses = np.array(state.reference_state.nuclear_masses)
    coords = np.array(state.reference_state.coordinates)
    coords = np.reshape(coords, (charges.size,3)) #reshape coordinates
    if state.reference_state.gauge_origin == 'mass_center':
        gauge_origin = np.einsum('i,ij->j', masses, coords)/masses.sum()
    elif state.reference_state.gauge_origin == 'charge_center':
        gauge_origin = np.einsum('i,ij->j ', charges, coords)/charges.sum()
    elif state.reference_state.gauge_origin == 'origin':
        gauge_origin = [0.0, 0.0, 0.0]
    elif type(state.reference_state.gauge_origin) == list:
        gauge_origin = state.reference_state.gauge_origin
    else:
        raise NotImplementedError("Gauge origin not correctly specified in adcc.")

    if op_type == "dia_magnet":
        op_int =  np.array(state.reference_state.operators.diag_mag) 
        size = op_int.shape[0]  
        coords = coords - gauge_origin
        r_r = np.einsum('ij,ik->ijk', coords, coords) # construct r*r matrix
        r_2 = np.zeros((charges.size, 3,3)) # construct r^2 
        for i in range(charges.size):
            for j in range(3):
                    r_2[i][j][j] = np.trace(r_r[i])
        term =  r_2 - r_r
        nuclear_gs = - 0.25 * np.einsum('i,i,ijk->jk', charges**2, 1/masses, term)
        nuclear_gs = 0 * nuclear_gs #no nuclear contribution needed
    elif op_type == "electric_quadrupole":
        op_int = -1.0 * np.array(state.reference_state.operators.electric_quadrupole) #electronic charge
        size = op_int.shape[0]
        nuc_gs = state.reference_state.nuclear_quadrupole #xx, xy, xz, yy, yz, zz
        nuclear_gs = np.zeros((3,3))
        nuclear_gs[0][0] = nuc_gs[0] #xx
        nuclear_gs[0][1] = nuclear_gs[1][0] = nuc_gs[1] #xy, yx
        nuclear_gs[0][2] = nuclear_gs[2][0] = nuc_gs[2]  #xz, zx
        nuclear_gs[1][1] = nuc_gs[3] #yy
        nuclear_gs[1][2] = nuclear_gs[2][1] = nuc_gs[4] #yz, zy
        nuclear_gs[2][2] = nuc_gs[5] #zz
    elif op_type == "electric_quadrupole_traceless":
        op_int = -1.0 * np.array(state.reference_state.operators.electric_quadrupole_traceless) #electronic charge
        size = op_int.shape[0]
        coords = coords - gauge_origin
        r_r = np.einsum('ij,ik->ijk', coords, coords) # construct r*r matrix
        r_2 = np.zeros((charges.size, 3,3)) # construct r^2 
        for i in range(charges.size):
            for j in range(3):
                    r_2[i][j][j] = np.trace(r_r[i])
        term =  3 * r_r - r_2
        nuclear_gs = 0.5 * np.einsum('i,ijk->jk', charges, term)
    else:
        raise NotImplementedError()

    pm_level = state.property_method.level
    ref_state_density = state.reference_state.density
    components = list(product(range(size), repeat = op_int.ndim))
    ref_state_moment = np.zeros((size,)*op_int.ndim)
    for c in components:
        ref_state_moment[c] = product_trace(op_int[c], ref_state_density) 
    if pm_level == 1:
        return  nuclear_gs + ref_state_moment
    elif pm_level == 2:
        mp2_corr = np.zeros((size,)*op_int.ndim)
        mp2_density = state.ground_state.mp2_diffdm
        for c in components:
            mp2_corr[c] = np.array(product_trace(op_int[c], mp2_density)) 
        return nuclear_gs + ref_state_moment + mp2_corr 
    elif pm_level == 3:
        mp3_corr = np.zeros((size,)*op_int.ndim)
        mp3_density = state.ground_state.mp3_diffdm
        for c in components:
            mp3_corr[c] = np.array(product_trace(op_int[c], mp3_density)) 
        return nuclear_gs + ref_state_moment + mp3_corr 
    else:
        raise NotImplementedError("Only dipole moments for level 1, 2, and 3"
                                      " are implemented.")

def transition_moments(state, op_type):
    assert op_type in available_operators
    if op_type == "dia_magnet":
        op_int= np.array(state.reference_state.operators.diag_mag)
        size = op_int.shape[0]
    elif op_type == "electric_quadrupole":
        op_int= np.array(state.reference_state.operators.electric_quadrupole)
        size = op_int.shape[0]
    elif op_type == "electric_quadrupole_traceless":
        op_int= np.array(state.reference_state.operators.electric_quadrupole_traceless)
        size = op_int.shape[0]
    elif op_type == "nabla":
        op_int= np.array(state.reference_state.operators.nabla)
        size = op_int.shape[0]
    else:
        raise NotImplementedError()
    if op_int.ndim == 2:
        tdms = np.zeros((state.size, size, size)
                )
    else:
        tdms = np.zeros((state.size, size)
                )
    components = list(product(range(size), repeat = op_int.ndim))
    for ee in tqdm(state.excitations):
        i = ee.index
        tdm = transition_dm(
            state.property_method,
            state.ground_state,
            ee.excitation_vector,
            state.matrix.intermediates,
            )
        for c in components:
            tdms[i][c] = np.array(product_trace(tdm, op_int[c]))
    return np.squeeze(tdms)

def state_to_state_transition_moments(state, op_type, initial_state=None, final_state=None):
    assert op_type in available_operators
    if isinstance(state, MockExcitedStatesTestData):
        if op_type == "electric":
            s2s_tdms = state.transition_dipole_moment_s2s
        elif op_type == "magnetic":
            s2s_tdms = state.transition_magnetic_moment_s2s
        elif op_type == "electric":
            s2s_tdms = state.transition_dipole_moment_s2s
        elif op_type == "dia_magnet":
            s2s_tdms = state.diag_mag_s2s
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
    elif  isinstance(state, MockExcitedStates):
        if op_type == "electric":
            s2s_tdms = state.s2s_transition_dipole_moment
        elif op_type == "magnetic":
            s2s_tdms = state.s2s_transition_magnetic_dipole_moment
        elif op_type == "nabla":
            s2s_tdms = state.s2s_nabla
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
        elif op_type == "dia_magnet":
            op_int= np.array(state.reference_state.operators.diag_mag)
            size = op_int.shape[0]
        elif op_type == 'electric_quadrupole':
            op_int = np.array(state.reference_state.operators.electric_quadrupole)
            size = op_int.shape[0]
        elif op_type == 'electric_quadrupole_traceless':
            op_int = np.array(state.reference_state.operators.electric_quadrupole_traceless)
            size = op_int.shape[0]
        elif op_type == "nabla":
            op_int = np.array(state.reference_state.operators.nabla)
            size = op_int.shape[0]
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
            if op_int.ndim == 2:
                s2s_tdms = np.zeros((state.size, 1, size, size))
            else:
                s2s_tdms = np.zeros((state.size, 1, size))
            excitations1 = state.excitations
            excitations2 = [state.excitations[final_state]]
        elif final_state is None:
            if op_int.ndim == 2:
                s2s_tdms = np.zeros((1, state.size, size, size))
            else:
                s2s_tdms = np.zeros((1, state.size, size))
            excitations1 = [state.excitations[initial_state]]
            excitations2 = state.excitations
        else:
            if op_int.ndim == 2:
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
                    s2s_tdms[i][j][c] = np.array(product_trace(tdm, op_int[c]))
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
            elif self._op_type == "dia_magnet":
                self._dips = self._state.reference_state.operators.diag_mag
            elif self._op_type == 'electric_quadrupole':
                self._dips = self._state.reference_state.operators.electric_quadrupole
            elif self._op_type == 'electric_quadrupole_traceless':
                self._dips = self._state.reference_state.operators.electric_quadrupole_traceless
            elif self._op_type == 'nabla':
                self._dips = self._state.reference_state.operators.nabla
            else:
                raise NotImplementedError()
        return self._dips

    @property
    def mtms(self):
        if self._mtms is None:
            if self._op_dim == 1:
                self._mtms = modified_transition_moments(
                        self._state.property_method, self._state.ground_state, self.dips
                        )
            elif self._op_dim == 2:
                self._mtms = [0, 0, 0] #initialize 
                for i in range(3):
                    self._mtms[i] =  modified_transition_moments(
                        self._state.property_method, self._state.ground_state, self.dips[i]
                        )
            else:
                raise NotImplementedError("Only operators up to two dimensions are implemented.")
        return self._mtms

    @property
    def gs_dip_moment(self):
        if self._gs_dip_moment is None:
            if isinstance(self._state, MockExcitedStatesTestData):
                pm_level = self._state.property_method.replace("adc", "")
                if self._op_type == "electric":
                    self._gs_dip_moment = self._state.ground_state.dipole_moment[pm_level]
                elif self._op_type == "magnetic":
                    self._gs_dip_moment = gs_magnetic_dipole_moment(self._state.ground_state, pm_level)
                else:
                    raise NotImplementedError()
            elif isinstance(self._state, MockExcitedStates):
                if self._op_type == "electric":
                    self._gs_dip_moment = self._state.dipole_moment
                elif self._op_type == "magnetic":
                    self._gs_dip_moment = self._state.magnetic_dipole_moment
                else:
                    raise NotImplementedError()
            else:
                pm_level = self._state.property_method.level
                if self._op_type == "electric":
                    self._gs_dip_moment = self._state.ground_state.dipole_moment(pm_level)
                elif self._op_type == "magnetic":
                    self._gs_dip_moment = gs_magnetic_dipole_moment(self._state.ground_state, pm_level)
                elif self._op_type in available_operators:
                    self._gs_dip_moment = ground_state_moments(self._state, self._op_type)
                else:
                    raise NotImplementedError()
        return self._gs_dip_moment 

    @property
    def transition_dipole_moment(self):
        if self._transition_dipole_moment is None:
            if isinstance(self._state, MockExcitedStatesTestData):
                if self._op_type == "electric":
                    self._transition_dipole_moment = self._state.transition_dipole_moment
                elif self._op_type == "magnetic":
                    self._transition_dipole_moment = self._state.transition_magnetic_dipole_moment
                elif self._op_type == "nabla":
                    self._transition_dipole_moment = self._state.transition_nabla
                else:
                    raise NotImplementedError()
            else:
                if self._op_type == "electric":
                    self._transition_dipole_moment = self._state.transition_dipole_moment
                elif self._op_type == "magnetic":
                    self._transition_dipole_moment = self._state.transition_magnetic_dipole_moment
                # elif self._op_type == "nabla":
                #     self._transition_dipole_moment = self._state.transition_dipole_moment_velocity
                elif self._op_type in available_operators:
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
    matrix = adcc.AdcMatrix("adc0", refstate)
    state = adcc.adc2(scfres, n_singlets=5)
    mp = state.ground_state
    
    adcc_prop = AdccProperties(state, "electric_quadrupole")
    #gs = adcc_prop.gs_dip_moment
    #print(gs)
    print(adcc_prop.mtms)
    print(type(adcc_prop.mtms))
    # tdms = adcc_prop.transition_dipole_moment
    # print(tdms)
    # s2s_tdms = adcc_prop.state_to_state_transition_moment
    # print(s2s_tdms)

    # print(adcc_prop.dips)


