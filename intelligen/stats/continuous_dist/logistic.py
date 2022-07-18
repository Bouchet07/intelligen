import numpy as np
import matplotlib.pyplot as plt

from ..ABCDistribution import ABCContinousDistribution, Distribution
from ...special import comb
from functools import cached_property

__all__ = ['Logistic', 'logistic', 'logit']

def logistic(x, x_0=0, L=1, k=1):
  """
  x_0, the x value of the sigmoid's midpoint;
  L, the curve's maximum value;
  k, the logistic growth rate or steepness of the curve
  """
  x = np.asarray(x)
  return L/(1+np.exp(-k*(x-x_0)))

def logit(p):
    p = np.asarray(p)
    return np.log(p/(1-p))

class Logistic(ABCContinousDistribution):

    def __init__(self, mu, s):
        self.mu, self.s = mu, s

    #---------Properties---------#
    @cached_property
    def mean(self): 
        return self.mu

    @cached_property
    def variance(self):
        return self.s**2 * np.pi**2 / 3
    
    @cached_property
    def skewness(self): 
        return 0

    @cached_property
    def kurtosis(self): 
        return 6/5
    
    @cached_property
    def entropy(self): 
        """
        Average level of "information", "surprise", or "uncertainty"
        inherent to the variable's possible outcomes
        """
        return np.log(self.s) + 2
    
    def pdf(self, x):
        temp = np.exp(-(x-self.mu)/self.s)
        return temp / (self.s*(1 + temp)**2)
    
    def cdf(self, x):
        # return 1/(1 + np.exp(-(x-self.mu)/self.s))
        return logistic(x, x_0=self.mu, k=1/self.s)

    def qf(self, p):
        return self.mu + self.s*logit(p)
    ppf = qf # for scipy consistency

    def qdf(self, p):
        p = np.asarray(p)
        return self.s/(p(1-p))

