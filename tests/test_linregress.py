import pytest
import numpy as np
from intelligen.linregress import LinearRegression, GradientDescent

def test_linear_regression():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    lr = LinearRegression()
    lr.fit(X, y)
    assert pytest.approx(lr.coef_(), abs=1e-3) == 2.0
    assert pytest.approx(lr.intercept_(), abs=1e-3) == 0.0

    preds = lr.predict(np.array([6]))
    assert pytest.approx(preds[0], abs=1e-3) == 12.0

def test_gradient_descent():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    gd = GradientDescent()
    gd.fit(X, y, epoch=100, lr=0.01)
    # The gradient descent might not perfectly reach 2.0 in 100 epochs, but it should be close
    assert gd.m > 1.5
    assert gd.b < 1.0

    preds = gd.predict(np.array([6]))
    assert preds[0] > 10.0
