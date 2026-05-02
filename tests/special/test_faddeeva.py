import pytest
import numpy as np
from intelligen.special._faddeeva import w, w_im, erf, erfc, erfi, erfcx, Dawson

# Precision settings
RTOL = 1e-13
ATOL = 1e-15

# --- Reference Data Sets ---

W_TESTS = [
    (0.0 + 0.0j, 1.0 + 0.0j),
    (1.0 + 1.0j, 0.30474420525691254 + 0.2082189382028316j),
    (-0.5 + 2.0j, 0.2452759902263585 - 0.0515214783436358j)
]

ERF_TESTS = [
    (0.0, 0.0),
    (1.0, 0.8427007929497148),
    (-0.5, -0.5204998778130465),
    (1.0 + 1.0j, 1.3161512816979477 + 0.19045346923783463j)
]

ERFCX_TESTS = [
    (0.0, 1.0),
    (1.0, 0.427583576155807),
    (1.0 + 1.0j, 0.30474420525691254 - 0.2082189382028316j)
]

DAWSON_TESTS = [
    (0.0, 0.0),
    (1.0, 0.5380795069127684),
    (1.0 + 1.0j, 0.9903730923223615 - 0.6388730515644433j)
]

W_IM_TESTS = [
    (0.0, 0.0),
    (1.0, 0.6071577058413937),
    (2.5, 0.25172302461185764)
]

# --- Scalar Tests ---

@pytest.mark.parametrize("z, expected", W_TESTS)
def test_w_scalar(z, expected):
    assert np.isclose(w(z), expected, rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize("z, expected", ERF_TESTS)
def test_erf_scalar(z, expected):
    assert np.isclose(erf(z), expected, rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize("z, expected", ERFCX_TESTS)
def test_erfcx_scalar(z, expected):
    assert np.isclose(erfcx(z), expected, rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize("z, expected", DAWSON_TESTS)
def test_dawson_scalar(z, expected):
    assert np.isclose(Dawson(z), expected, rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize("x, expected", W_IM_TESTS)
def test_w_im_scalar(x, expected):
    assert np.isclose(w_im(x), expected, rtol=RTOL, atol=ATOL)

# --- Functional & Array Tests ---

def test_erfi_logic():
    # erfi(z) = -i * erf(iz)
    # Using 1+1j as input
    z = 1.0 + 1.0j
    expected = 0.19045346923783463 + 1.3161512816979477j
    assert np.isclose(erfi(z), expected, rtol=RTOL)

def test_vectorization_and_shape():
    """Verifies apply_func handles multi-dimensional arrays and parallel dispatch."""
    data = np.array([[0.0, 1.0], [-0.5, 2.5]])
    expected = np.array([
        [0.0, 0.8427007929497148], 
        [-0.5204998778130465, 0.999593047982555]
    ])
    
    res = erf(data)
    assert res.shape == (2, 2)
    assert np.allclose(res, expected, rtol=RTOL)

def test_const_input_support():
    """Checks if the ndarray<const T> binding allows read-only numpy arrays."""
    arr = np.array([1.0, 0.0])
    arr.flags.writeable = False
    # This should execute without throwing a buffer error
    res = erfc(arr)
    assert np.isclose(res[0], 0.15729920705028516)