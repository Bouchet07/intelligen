import pytest
from intelligen.stats.discrete_dist.bernoulli import Bernoulli

def test_bernoulli():
    b = Bernoulli(0.65)
    assert pytest.approx(b.pmf(0), abs=1e-3) == 0.35
    assert pytest.approx(b.pmf(1), abs=1e-3) == 0.65
    assert pytest.approx(b.mean, abs=1e-3) == 0.65
    assert pytest.approx(b.variance, abs=1e-3) == 0.65 * 0.35
