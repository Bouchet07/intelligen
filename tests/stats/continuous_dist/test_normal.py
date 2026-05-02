import pytest
from intelligen.stats.continuous_dist.normal import Normal

def test_normal_basic():
    dist = Normal(0, 1)
    assert dist.mean == 0
    assert dist.variance == 1
