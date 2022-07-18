import numpy as np
import matplotlib.pyplot as plt

from ..ABCDistribution import ABCDiscreteDistribution, Distribution
from ...special.typings import *
from .binomial import Binomial
from functools import cached_property

__all__ = ['Bernoulli']

class Bernoulli(ABCDiscreteDistribution):
    """
    Bernoulli Distribution
    ======================

    Discrete probability distribution of a random variable
    which takes the value 1 with probability`p`and the
    value 0 with probability `q = 1-p`

    Parameters
    ----------
    p : Real
        Probability of success
    
    Attributes
    ----------
    q : Real
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
    >>> from intelligen.stats import Bernoulli
    >>> B = Bernoulli(0.65)
    >>> B.pmf(0)
    0.35
    >>> B.skewness
    -0.6289709020331511

    """
    def __init__(self, p: Real) -> None:
        if not 0<=p<=1: raise ValueError(f'{p = } must be a valid probability between 0-1')
        self.p, self.q = p, 1-p

    #---------Properties---------#
    @cached_property
    def mean(self): 
        return self.p

    @cached_property
    def variance(self):
        return self.p * self.q
    
    @cached_property
    def skewness(self): 
        return (self.q - self.p) / np.sqrt(self.variance)

    @cached_property
    def kurtosis(self): 
        return (1 - 6*self.variance) / self.variance
    
    @cached_property
    def entropy(self): 
        """
        Average level of "information", "surprise", or "uncertainty"
        inherent to the variable's possible outcomes
        """
        return -self.q*np.log(self.q) -self.p*np.log(self.p)

    def __add__(self, distribution: Distribution) -> Distribution:
        if isinstance(distribution, Bernoulli):
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
        return f'Bernoulli(p={self.p})'
        

    def pmf(self, k: Union[int,Vector_int]) -> Union[Real,Vector]:
        if isinstance(k, vector):
            # return np.array([self.pmf(ki) for ki in k])
            k = np.asarray(k, dtype=float)

            if np.any((k!=0)&(k!=1)): return self.p**k * self.q**(1-k)#return Binomial(1, self.p).pmf(k)#raise ValueError('R{0,1}')
            k[k==0] = self.q
            k[k==1] = self.p
            return k
            
        else:
            if   k == 0: return self.q
            elif k == 1: return self.p
            else: return Binomial(1, self.p).pmf(k)#raise ValueError('R{0,1}')

    def cdf(self, k: Union[Real,Vector]) -> Union[Real,Vector]:
        if isinstance(k, vector):
            # return np.array([self.cdf(ki) for ki in k])
            k = np.asarray(k, dtype=float)
            
            k[(0<=k)&(k<1)] = self.q
            k[k<0] = 0
            k[k>=1] = 1

            return k

        else:
            if   k < 0: return 0
            elif k < 1: return self.q
            else: return 1

    def pdf(self, a: Union[Real,Vector], b: Union[Real,Vector]) -> Union[Real,Vector,Matrix2D]:
        if isinstance(a, vector):
            if isinstance(b, vector):
                return np.tile(self.cdf(b).reshape(-1,1), len(a)) - self.cdf(a)
        
        return self.cdf(b) - self.cdf(a)

    def mgf(self, t: Union[Real,Vector]) -> Union[Real,Vector]:
        return self.q + self.p*np.exp(t)

    def cf(self, t: Union[Real,Vector]) -> Union[Real,Vector]:
        return self.q + self.p*np.exp(1j*t)
    
    def rmoment(self, order: Union[int, Vector_int]) -> Union[Real,Vector]:
        if isinstance(order, vector): return np.full_like(order, self.p, dtype=float)
        return self.p

    def cmoment(self, order: Union[int, Vector_int]) -> Union[Real,Vector]:
        return self.q*np.power(-self.p, order) + self.p*np.power(self.q, order)

    #-----------Plots-----------#
    def plot_pmf(self, ax=None) -> plt.axes:
        if ax is None: ax = plt.gca()

        ax.set_title(f'Probability mass function\nBernoulli')
        ax.set_ylabel('Probability')

        bar_plot = ax.bar([0,1],self.pmf([1,0]), width=0.3, tick_label = ['p','1-p'])
        ax.bar_label(bar_plot,[f'{self.p:.3f}',f'{self.q:.3f}'])
        # ax.get_xaxis().set_visible(False)
        return ax
    
    def plot_cdf(self, ax=None) -> plt.axes:
        if ax is None: ax = plt.gca()
        ax.set_title(f'Cumulative distribution function\nBernoulli')
        ax.step([-1,0,1,2],self.cdf([-1,0,1,2]), where='post')
        return ax
    
    def plot_pdf(self, ax=None) -> plt.axes:
        if ax is None: ax = plt.gca()
        
        return ax