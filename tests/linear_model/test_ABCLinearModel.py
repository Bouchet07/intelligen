import pytest
from intelligen.linear_model.ABCLinearModel import LinearModel
import numpy as np

class DummyModel(LinearModel):
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.coef_ = np.array([[2.0]])
        self.intercept_ = 0.0
        self._fitted = True

def test_abc_linear_model():
    model = DummyModel()
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    model.fit(X, y)
    
    preds = model.predict(X)
    assert pytest.approx(preds[0][0], abs=1e-3) == 2.0
    assert pytest.approx(preds[1][0], abs=1e-3) == 4.0
