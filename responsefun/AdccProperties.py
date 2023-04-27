#  Copyright (C) 2023 by the responsefun authors
#
#  This file is part of responsefun.
#
#  responsefun is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  responsefun is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with responsefun. If not, see <http:www.gnu.org/licenses/>.
#

import warnings
from itertools import product

import numpy as np
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.OneParticleOperator import product_trace
from cached_property import cached_property
from tqdm import tqdm

from responsefun.testdata.mock import MockExcitedStates
from responsefun.transition_dm import transition_dm

# dict of operators available in responsefun so far
# the first argument specifies the symbol that is to be used for printing
# the second argument specifies the symmetry:
#   0: no symmetry assumed
#   1: hermitian
#   2: anti-hermitian
# the third argument specifies the dimensionality
available_operators = {
        "electric": ("mu", 1, 1),
        "magnetic": ("m", 2, 1),
        "dia_magnet": ("xi", 1, 2),
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
def transition_moments(state, operator):
    if state.property_method.level == 0:
        warnings.warn("ADC(0) transition moments are known to be faulty in some cases.")

    op_shape = np.shape(operator)
    iterables = [list(range(shape)) for shape in op_shape]
    components = list(product(*iterables))
    moments = np.zeros((state.size, *op_shape))
    for i, ee in enumerate(tqdm(state.excitations)):
        tdm = transition_dm(state.property_method, state.ground_state, ee.excitation_vector)
        tms = np.zeros(op_shape)
        for c in components:
            # list indices must be integers (1-D operators)
            c = c[0] if len(c) == 1 else c
            tms[c] = product_trace(operator[c], tdm)
        moments[i] = tms
    return np.squeeze(moments)


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
            tms = np.zeros(op_shape)
            for c in components:
                # list indices must be integers (1-D operators)
                c = c[0] if len(c) == 1 else c
                tms[c] = product_trace(tdm, operator[c])
            s2s_tm[i, j] = tms
    return np.squeeze(s2s_tm)


# TODO: testing
def gs_magnetic_dipole_moment(ground_state, level=2):
    magdips = ground_state.reference_state.operators.magnetic_dipole
    ref_dipmom = np.array(
        [product_trace(dip, ground_state.reference_state.density) for dip in magdips]
    )
    if level == 1:
        return ref_dipmom
    elif level == 2:
        mp2corr = - 1.0 *  np.array(
                [product_trace(dip, ground_state.mp2_diffdm) for dip in magdips]
        )
        return ref_dipmom + mp2corr
    else:
        raise NotImplementedError(
            "Only magnetic dipole moments for level 1 and 2" " are implemented."
        )


class AdccProperties:
    """Class encompassing all properties that can be obtained from adcc for a given operator."""

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
            return np.array(self._state.reference_state.operators.magnetic_dipole)
        elif self._op_type == "dia_magnet":
            return np.array(self._state.reference_state.operators.diag_mag)
        elif self._op_type == "electric_quadrupole":
            return np.array(self._state.reference_state.operators.electric_quadrupole)
        elif self._op_type == "electric_quadrupole_traceless":
            return np.array(self._state.reference_state.operators.electric_quadrupole_traceless)
        elif self._op_type == "nabla":
            return np.array(self._state.reference_state.operators.nabla)
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
            elif self._op_type in available_operators:
                gs_moment = ground_state_moments(self._state, self.op_type)
            else:
                raise NotImplementedError()
        return gs_moment

    @cached_property
    def transition_moment(self):
        if self.op_type == "electric":
            return self._state.transition_dipole_moment
        elif self._op_type == "nabla":
            return self._state.transition_dipole_moment_velocity
        # TODO: use commented code once PR #158 of adcc has been merged
        # elif self.op_type == "magnetic":
        #     return self._state.transition_magnetic_dipole_moment
        else:
            if isinstance(self._state, MockExcitedStates):
                if self.op_type == "magnetic":
                    return self._state.transition_magnetic_dipole_moment
                elif self.op_type == "electric_quadrupole":
                    return self._state.transition_electric_quadrupole_moment
                else:
                    raise NotImplementedError()
            return transition_moments(self._state, self.operator)

    @cached_property
    def state_to_state_transition_moment(self):
        if isinstance(self._state, MockExcitedStates):
            if self.op_type == "electric":
                return self._state.transition_dipole_moment_s2s
            elif self.op_type == "magnetic":
                return self._state.transition_magnetic_moment_s2s
            elif self.op_type == "nabla":
                return self._state.transition_nabla_s2s
            elif self.op_type == "electric_quadrupole":
                return self._state.transition_electric_quadrupole_moment_s2s
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

