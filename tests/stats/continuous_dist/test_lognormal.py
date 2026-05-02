import pytest
from intelligen.stats.continuous_dist.lognormal import LogNormal

def test_lognormal_basic():
    dist = LogNormal(0, 1)
    assert dist.mu == 0
    assert dist.s == 1
