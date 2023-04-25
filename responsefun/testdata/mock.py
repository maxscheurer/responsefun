import numpy as np


class MockExcitedStates:
    """Mock class for excited states based on zarr file"""
    def __init__(self, zr):
        self.zr = zr
        exci = self.zr.excitation
        for k in exci.attrs:
            setattr(self, k, exci.attrs[k])
        for k in exci:
            setattr(self, k,
                    np.asarray(exci[k]))
        self.ground_state = self.zr.ground_state