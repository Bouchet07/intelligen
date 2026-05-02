import pytest
from intelligen.stats.discrete_dist.binomial import Binomial

def test_binomial_basic():
    dist = Binomial(10, 0.5)
    assert dist.mean == 5.0
    assert dist.variance == 2.5
