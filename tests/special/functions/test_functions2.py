import pytest
from intelligen.special.functions.functions2 import factorial, comb, fibonacci, binet

def test_factorial():
    assert factorial(0) == 1
    assert factorial(5) == 120
    with pytest.raises(ValueError):
        factorial(-1)

def test_comb():
    assert comb(5, 2, exact=True) == 10
    assert comb(5, 0, exact=True) == 1
    assert comb(5, 6) == 0

def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5
    assert fibonacci(10) == 55
    assert fibonacci(5, list=True) == [0, 1, 1, 2, 3, 5]

def test_binet():
    assert pytest.approx(binet(10), abs=1e-3) == 55
