import pytest
from intelligen import numeric

def f(x): return x**3 + 2*x**2 + 10*x - 20
ROOT = 1.36880

def test_newton():
    z = numeric.newton(f, 1, 0.01, True)[0]
    assert pytest.approx(z, abs=1e-3) == ROOT

def test_bisection():
    z = numeric.bisection(f, 1, 2, 0.01, True)[0]
    assert pytest.approx(z, abs=1e-3) == ROOT

def test_regula_falsi():
    z = numeric.regula_falsi(f, 1, 2, 0.01, True)[0]
    assert pytest.approx(z, abs=1e-3) == ROOT

def test_secant():
    z = numeric.secant(f, 1, 2, 0.01, True)[0]
    assert pytest.approx(z, abs=1e-3) == ROOT

def test_newton2():
    z = numeric.newton2(f, 1, 0.01, True)[0]
    assert pytest.approx(z, abs=1e-3) == ROOT
