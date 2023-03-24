# taken from respondo

import adcc
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.OneParticleOperator import product_trace
from static_data import xyz
from cache import cases

import zarr
import numpy as np
from tqdm import tqdm

from responsefun.AdccProperties import transition_moments


def main():
    for case in cases:
        n_singlets = cases[case]
        molecule, basis, method = case.split("_")
        scfres = adcc.backends.run_hf(
            "pyscf", xyz=xyz[molecule],
            basis=basis,
            # conv_tol=conv_tol,
            # multiplicity=multiplicity,
            # conv_tol_grad=conv_tol_grad,
        )
        state = adcc.run_adc(method=method, data_or_matrix=scfres,
                             n_singlets=n_singlets)
        dips = state.reference_state.operators.electric_dipole
        mdips = state.reference_state.operators.magnetic_dipole

        # state to state transition moments
        s2s_tdms = np.zeros((state.size, state.size, 3))
        s2s_tdms_mag = np.zeros((state.size, state.size, 3))
        for ee1 in tqdm(state.excitations):
            i = ee1.index
            for ee2 in state.excitations:
                j = ee2.index
                tdm = state2state_transition_dm(
                    state.property_method,
                    state.ground_state,
                    ee1.excitation_vector,
                    ee2.excitation_vector,
                    state.matrix.intermediates,
                )
                tdm_fn = np.array([product_trace(tdm, dip) for dip in dips])
                tdm_mag = np.array([product_trace(tdm, mdip) for mdip in mdips])
                s2s_tdms[i, j] = tdm_fn
                s2s_tdms_mag[i, j] = tdm_mag

        z = zarr.open(f'{case}.zarr', mode='w')
        z.create_group('excitation')
        exci = z['excitation']
        propkeys = state.excitation_property_keys
        propkeys.extend([k.name for k in state._excitation_energy_corrections])
        for key in propkeys:
            try:
                d = getattr(state, key)
            except NotImplementedError:
                continue
            if not isinstance(d, np.ndarray):
                continue
            if not np.issubdtype(d.dtype, np.number):
                continue
            exci[key] = d

        # TODO: remove line once PR #158 of adcc has been merged
        exci['transition_magnetic_dipole_moment'] = transition_moments(state, mdips)
        exci['transition_dipole_moment_s2s'] = s2s_tdms
        exci['transition_magnetic_moment_s2s'] = s2s_tdms_mag
        exci.attrs['kind'] = state.kind
        exci.attrs['method'] = state.method.name
        exci.attrs['property_method'] = state.property_method.name
        mp = state.ground_state
        hf = state.reference_state
        z['ground_state/dipole_moment/1'] = mp.dipole_moment(1)
        z['ground_state/dipole_moment/2'] = mp.dipole_moment(2)
        z['ground_state/energy/2'] = mp.energy(2)
        z['ground_state/energy/3'] = mp.energy(3)
        z['reference_state/energy_scf'] = hf.energy_scf
        z['reference_state/dipole_moment'] = hf.dipole_moment


if __name__ == "__main__":
    main()
