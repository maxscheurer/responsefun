# taken from respondo

import os
from .mock import MockExcitedStates


cases = {
    # "h2o_sto3g_adc1": 10,
    "h2o_sto3g_adc2": 65,
    # "h2o_ccpvdz_adc1": 95,
    # "h2o_ccpvdz_adc2": 4655,
    # "formaldehyde_sto3g_adc1": 32,
    # "formaldehyde_sto3g_adc2": 560,
}


def read_full_diagonalization():
    import zarr
    ret = {}
    for case in cases:
        thisdir = os.path.dirname(__file__)
        zarr_file = os.path.join(thisdir, f"{case}.zarr")
        if not os.path.isdir(zarr_file):
            continue
        z = zarr.open(zarr_file, mode='r')
        ret[case] = MockExcitedStates(z)
    return ret


data_fulldiag = read_full_diagonalization()
