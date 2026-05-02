import pytest
import numpy as np
from intelligen.stats.continuous_dist.logistic import Logistic, logistic, logit

def test_logistic_functions():
    assert pytest.approx(logistic(0), abs=1e-3) == 0.5
    assert pytest.approx(logit(0.5), abs=1e-3) == 0.0

def test_logistic_class():
    dist = Logistic(mu=0, s=1)
    assert dist.mean == 0
    assert pytest.approx(dist.variance, abs=1e-3) == np.pi**2 / 3
    assert pytest.approx(dist.cdf(0), abs=1e-3) == 0.5
