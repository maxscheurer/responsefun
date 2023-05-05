from adcc import AmplitudeVector
from adcc.IsrMatrix import IsrMatrix
from respondo.cpp_algebra import ResponseVector as RV


def scalar_product(left_v, right_v):
    """Evaluate the scalar product between two instances of ResponseVector and/or
    AmplitudeVector."""
    if isinstance(left_v, AmplitudeVector):
        lv = RV(left_v)
    else:
        lv = left_v.copy()
    if isinstance(right_v, AmplitudeVector):
        rv = RV(right_v)
    else:
        rv = right_v.copy()
    assert isinstance(lv, RV) and isinstance(rv, RV)
    real = lv.real @ rv.real - lv.imag @ rv.imag
    imag = lv.real @ rv.imag + lv.imag @ rv.real
    if imag == 0:
        return real
    else:
        return real + 1j * imag


# TODO: testing
def bmatrix_vector_product(bmatrix, rvec):
    assert isinstance(bmatrix, IsrMatrix)
    assert isinstance(rvec, RV)
    product_real = bmatrix @ rvec.real
    product_imag = bmatrix @ rvec.imag

    unpack = False
    if not isinstance(product_real, list):
        assert not isinstance(product_imag, list)
        unpack = True
        product_real = [product_real]
        product_imag = [product_imag]
    
    ret = []
    for real, imag in zip(product_real, product_imag):
        ret.append(RV(real, imag))
    
    if unpack:
        assert len(ret) == 1
        ret = ret[0]
    return ret