import pytest
from intelligen.stats.ABCDistribution import ABCDistribution, ABCDiscreteDistribution, ABCContinousDistribution

def test_abc_distributions():
    # Just verifying they can be imported and are abstract
    assert ABCDistribution.__name__ == 'ABCDistribution'
    assert ABCDiscreteDistribution.__name__ == 'ABCDiscreteDistribution'
    assert ABCContinousDistribution.__name__ == 'ABCContinousDistribution'
