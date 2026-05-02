import pytest
import numpy as np
from intelligen.linear_model.linear_regression import LinearRegression

def test_linear_regression_fit():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    
    # Using mine implementation
    """ lr = LinearRegression(implementation='mine')
    lr.fit(X, y)
    assert pytest.approx(lr.coef_[0][0], abs=1e-3) == 2.0
    assert pytest.approx(lr.intercept_, abs=1e-3) == 0.0 """

    # Using numpy implementation
    lr_np = LinearRegression(implementation='numpy')
    lr_np.fit(X, y)
    assert pytest.approx(lr_np.coef_[0], abs=1e-3) == 2.0
    assert pytest.approx(lr_np.intercept_, abs=1e-3) == 0.0
