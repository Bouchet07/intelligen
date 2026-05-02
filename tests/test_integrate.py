import pytest
import numpy as np
from intelligen import integrate

def test_trapz():
    N, a, b = 12, 1, 4
    x = np.linspace(a, b, N+1)
    y = np.sin(x)
    assert pytest.approx(integrate.trapz(y, x), abs=1e-5) == 1.18772

def test_simpson():
    N, a, b = 12, 1, 4
    x = np.linspace(a, b, N+1)
    y = np.sin(x)
    assert pytest.approx(integrate.simpson(y, x), abs=1e-5) == 1.19397

def test_simpson3_8():
    N, a, b = 12, 1, 4
    x = np.linspace(a, b, N+1)
    y = np.sin(x)
    assert pytest.approx(integrate.simpson3_8(y, x), abs=1e-5) == 1.19400

def test_odeEuler():
    def q(t): return 1/2 + 1/2 * np.cos(t**2)
    def f(t, y): return q(t) - y
    T, U = integrate.odeEuler(f, 0, 6, 100, 0)
    assert len(T) == 101
    assert len(U) == 101
