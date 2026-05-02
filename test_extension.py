import intelligen.special as sp
import numpy as np

# Test scalar real
print("erf(1.0) =", sp.erf(1.0))

# Test scalar complex
print("erf(1.0+1.0j) =", sp.erf(1.0+1j))

# Test numpy array float64 (double)
x_float = np.array([1.0, 2.0, 3.0], dtype=np.float64)
print("erf(array float64) =", sp.erf(x_float))

# Test numpy array complex128 (std::complex<double>)
x_complex = np.array([1.0+1j, 2.0, 3.0], dtype=np.complex128)
print("erf(array complex128) =", sp.erf(x_complex))

# Test erfinv array and scalar
print("erfinv(0.5) =", sp.erfinv(0.5))
x_erfinv = np.array([0.1, 0.5, 0.9], dtype=np.float64)
print("erfinv(array) =", sp.erfinv(x_erfinv))

print("All tests passed!")