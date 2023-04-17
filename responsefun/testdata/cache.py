# taken from respondo

import os
import numpy as np
import zarr


cases = {
    # "h2o_sto3g_adc1": 10,
    "h2o_sto3g_adc2": 65,
    # "h2o_ccpvdz_adc1": 95,
    # "h2o_ccpvdz_adc2": 4655,
    # "formaldehyde_sto3g_adc1": 32,
    # "formaldehyde_sto3g_adc2": 560,
}


class MockExcitedStates:
    def __init__(self, zr):
        self.zr = zr
        exci = self.zr.excitation
        for k in exci.attrs:
            setattr(self, k, exci.attrs[k])
        for k in exci:
            setattr(self, k,
                    np.asarray(exci[k]))
        self.ground_state = self.zr.ground_state


def read_full_diagonalization():
    ret = {}
    for case in cases:
        thisdir = os.path.dirname(__file__)
        zarr_file = os.path.join(thisdir, f"{case}.zarr")
        if not os.path.isdir(zarr_file):
            continue
        z = zarr.open(zarr_file, mode='r')
        ret[case] = MockExcitedStatesTestData(z)
    return ret


data_fulldiag = read_full_diagonalization()
