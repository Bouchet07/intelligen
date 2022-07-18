import numpy as np
import matplotlib.pyplot as plt

from ..ABCDistribution import ABCDiscreteDistribution, Distribution
from ...special.typings import *
from ...special import comb
from functools import cached_property

__all__ = ['Binomial']

class Binomial(ABCDiscreteDistribution):
    """
    Binomial Distribution
    =====================

    Discrete probability distribution that models the number
    of successes in a sequence of `n` independent experiments
    (Bernoulli trials) with constant probability `p`

    Parameters
    ----------
    n: int
        Number of Bernoulli trials
    
    p : Real
        Probability of success

    Attributes
    ----------
    q : float
        `1-p`

    mean : Real
        Expected value / mean 

    variance : Real
        Measure of dispersion

    skewness : Real
        Measure of the asymmetry

    kurtosis: Real
        Measure of the "tailedness"

    Examples
    --------
    >>> from intelligen.stats import Binomial
    >>> B = Binomial(17, 0.65)
    >>> B.pmf(10)
    0.1684553555671748
    >>> B.kurtosis
    -0.09437621202327084
    """

    def __init__(self, n: int, p: Real) -> None:
        self.n, self.p, self.q = n, p, 1-p

    #---------Properties---------#
    @cached_property
    def mean(self): 
        return self.n * self.p

    @cached_property
    def variance(self):
        return self.n * self.p * self.q
    
    @cached_property
    def skewness(self): 
        return (self.q - self.p) / np.sqrt(self.variance)

    @cached_property
    def kurtosis(self): 
        return (1 - 6*self.p * self.q) / self.variance
    

    def __add__(self, distribution: Distribution) -> Distribution:
        if isinstance(distribution, self.__class__):
            if self.p == distribution.p:
                return Binomial(2, self.p)
            else: raise ValueError('Probability must be the same')
        
        elif isinstance(distribution, Binomial):
            if self.p == distribution.p:
                return Binomial(distribution.n + 1, self.p)
            else: raise ValueError('Probability must be the same')
        
        else: raise ValueError('Distributions must be compatible')
    
    def __radd__(self, distribution: Distribution) -> Distribution:
        return self + distribution
    
    def __mul__(self, coef: integer) -> Distribution:
        if isinstance(coef, integer) : return Binomial(coef, self.p)
        else: raise ValueError('The coeficient must be an integer')
    
    def __rmul__(self, coef: integer) -> Distribution:
        return self * coef
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'
        

    def pmf(self, k: Union[int,Vector_int]) -> Union[Real,Vector]:
        if isinstance(k, vector): k = np.asarray(k)
            
        return comb(self.n, k) * self.p**k * self.q**(self.n-k)

    def cdf(self, k: Union[Real,Vector]) -> Union[Real,Vector]:
        if isinstance(k, vector):
            return np.array([self.cdf(i) for i in k])
        else:
            if   k < 0: return 0
            elif k < 1: return self.q
            else: return 1

    def pdf(self, a: Real, b: Real) -> Real:
        return self.cdf(b) - self.cdf(a)

    def plot_pmf(self, ax=None):
        if ax is None: ax = plt.gca()
        ax.set_title(f'Probability mass function\nBernoulli')
        bar_plot = ax.bar([0,1],self.pmf([1,0]), width=0.4)
        ax.bar_label(bar_plot,['p','1-p'])
        return ax
    
    def plot_cdf(self, ax=None):
        if ax is None: ax = plt.gca()
        ax.set_title(f'Cumulative distribution function\nBernoulli')
        ax.step([-1,0,1,2],self.cdf([-1,0,1,2]), where='post')
        return ax
    
    def plot_pdf(self, ax=None) -> plt.axes:
        return super().plot_pdf(ax)