import numpy as np
from ..ABCDistribution import ABCContinousDistribution, Distribution
from ...special import erf, erfinv
from functools import cached_property

__all__ = ['Normal']

class Normal(ABCContinousDistribution):
    """
    Normal Distribution
    ===================

    Parameters
    ----------
    mu: float
        mean of the distribution (\mu), default = 0
    s: floart
        standard deviation (\sigma), default = 1
    """
    def __init__(self, mu: float = 0, s: float = 1) -> None:
        self.mu, self.s = mu, s
    
    #---------Properties---------#
    @cached_property
    def mean(self): 
        return self.mu

    @cached_property
    def variance(self):
        return self.s**2
    
    @cached_property
    def skewness(self): 
        return 0

    @cached_property
    def kurtosis(self): 
        return 0
    
    @cached_property
    def entropy(self): 
        """
        Average level of "information", "surprise", or "uncertainty"
        inherent to the variable's possible outcomes
        """
        return 1/2*np.log(2*np.pi*self.s**2) + 1/2

    def pdf(self, x):
        return np.exp((-1/2) * ((x - self.mu)/self.s)**2) / (self.s * np.sqrt(2*np.pi))
    
    def cdf(self, k):
        k = np.asarray(k)
        return 1/2 * (1+erf((k-self.mu)/(self.s*np.sqrt(2))))
    
    def qf(self, p):
        p = np.asarray(p)
        return self.mu + self.s*np.sqrt(2)*erfinv(2*p-1)
    ppf = qf
