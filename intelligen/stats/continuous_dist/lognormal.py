import numpy as np
from ..ABCDistribution import ABCContinousDistribution, Distribution
from ...special import erf, erfinv
from functools import cached_property

__all__ = ['LogNormal']

class LogNormal(ABCContinousDistribution):

    def __init__(self, mu: float = 0, s: float = 1) -> None:
        self.mu, self.s = mu, s
    
    #---------Properties---------#
    @cached_property
    def mean(self): 
        return np.exp(self.mu + self.s**2/2)

    @cached_property
    def variance(self):
        return (np.exp(self.s**2)-1)*np.exp(2*self.mu + self.s**2)
    
    @cached_property
    def skewness(self): 
        return (np.exp(self.s**2)+2)*np.sqrt(np.exp(self.s**2)-1)

    @cached_property
    def kurtosis(self): 
        return np.exp(4*self.s**2) + 2*np.exp(3*self.s**2) + 3*np.exp(2*self.s**2) - 6
    
    @cached_property
    def entropy(self): 
        """
        Average level of "information", "surprise", or "uncertainty"
        inherent to the variable's possible outcomes
        """
        return np.log2(self.s*np.exp(self.mu+1/2)*np.sqrt(2*np.pi))

    def pdf(self, x):
        return np.exp(-(np.log(x)-self.mu)**2/(2*self.s**2)) / (x*self.s * np.sqrt(2*np.pi))
    
    def cdf(self, k):
        k = np.asarray(k)
        return 1/2 * (1+erf((np.log(k)-self.mu)/(self.s*np.sqrt(2))))
    
    def qf(self, p):
        p = np.asarray(p)
        return np.exp(self.mu + self.s*np.sqrt(2)*erfinv(2*p-1))
    ppf = qf
