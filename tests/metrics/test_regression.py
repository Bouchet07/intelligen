import pytest
import numpy as np
from intelligen.metrics.regression import mean_squared_error, mean_absolute_error, r2_score

def test_mse():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    assert pytest.approx(mean_squared_error(y_true, y_pred), abs=1e-3) == 0.375

def test_mae():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    assert pytest.approx(mean_absolute_error(y_true, y_pred), abs=1e-3) == 0.5

def test_r2_score():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    assert r2_score(y_true, y_pred) > 0.9
