import pandas as pd
import numpy as np
import re
import zarr
from responsefun.methods import *


class MockExcitedState:

    def __init__(self, method, *args):

        """
        Parameters: 
        method: str
            read_only --> nur zarr file auslesen
            qchem_tddft 
            qchem_adc
            qchem_ccsd
            dalton_cis

        *args: str
            outfile 1 und ggf outfile 2 als string geben

        returns:
        MockExcitedState
        """

        if  method =='read_only':
            zarr_storage = f'{args[0]}.zarr'
            print(f'mode: read only\nreading {args[0]}.zarr')

        elif method =='qchem_tddft':
            zarr_storage = qchem_read_tddft(args[0])

        elif method == 'qchem_adc':
            zarr_storage = qchem_read_adc(args[0])

        elif method == 'qchem_ccsd':
            zarr_storage = qchem_read_ccsd(args[0], args[1])

        elif method == 'qchem_fci':
            zarr_storage =  qchem_read_fci(args[0], args[1])

        elif method == 'dalton_cis':
            zarr_storage = read_dalton_cis(args[0],args[1])
        else:
            return NotImplementedError()

        #print(f'out file data stored in {zarr_storage}')

        if isinstance(zarr_storage, str):
            zr = zarr.open(zarr_storage, mode = 'r')
            self.zr = zr
            exci = self.zr.excited_state
            gs = self.zr.ground_state
            for k in exci.attrs:
                setattr(self, k , exci.attrs[k])
            for k in exci:
                setattr(self, k, np.asarray(exci[k]))
            try:
                self.s2s_transition_dipole_moment  = np.asarray(self.zr.s2s_transition_dipole_moment)
            except AttributeError:
                pass
            try:
                self.s2s_transition_magnetic_dipole_moment = np.asarray(self.zr.s2s_transition_magnetic_dipole_moment)
            except AttributeError:
                pass
            for k in gs.attrs:
                setattr(self, k , gs.attrs[k])
            for k in gs:
                setattr(self, k, np.asarray(gs[k]))

            setattr(self, 'excitation_energy_uncorrected', np.asarray(self.zr.excitation_energy_uncorrected))
            #self.excitation_energy_uncorrected = self.zr.excitation_energy_uncorrected
            #self.ground_state = self.zr.ground_state




if __name__ == "__main__":

    out_datei = '../../../Dalton/CIS_CO/cis_co.out'
    #out_datei_2 = 'test_h2o_3.out'
    #zarr_storage = qchem_read_ccsd(out_datei,out_datei_2, 10 )
    #print(zarr_storage)
    #y = groundstate(ground_state)
    x = MockExcitedState('dalton_cis', out_datei, None)
    #print(x.gs.energy)
    #print(x.s2s_transition_magnetic_dipole_moment)
    print(x.excitation_energy_uncorrected)
    #print(x.ground_state.dipole_moment)

