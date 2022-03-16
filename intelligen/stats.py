import numpy as np
import matplotlib.pyplot as plt
#from math import factorial
from typing import List
Vector = List[float]

from numpy import pi, sqrt
from numeric import factorial, combination
from integrate import trapz
from math import erf


#---------------------- Methods -----------------------#
                
def erf2(x: float) -> float:
    def integrand(t): return np.exp(-t**2)
    return 2/sqrt(pi) * trapz(f=integrand, a=0, b=x, N=100)

def erf4(x):
    """ Scipy implementation """
    c = .564189583547756e0
    a = (.771058495001320e-04, -.133733772997339e-02,
         .323076579225834e-01, .479137145607681e-01,
         .128379167095513e+00)
    b = (.301048631703895e-02, .538971687740286e-01,
         .375795757275549e+00)
    p = (-1.36864857382717e-07 ,5.64195517478974e-01,
         7.21175825088309e+00 ,4.31622272220567e+01,
         1.52989285046940e+02 ,3.39320816734344e+02,
         4.51918953711873e+02 ,3.00459261020162e+02)
    q = (1.00000000000000e+00, 1.27827273196294e+01,
         7.70001529352295e+01, 2.77585444743988e+02,
         6.38980264465631e+02, 9.31354094850610e+02,
         7.90950925327898e+02, 3.00459260956983e+02)
    r = (2.10144126479064e+00, 2.62370141675169e+01,
         2.13688200555087e+01, 4.65807828718470e+00,
         2.82094791773523e-01)
    e = (9.41537750555460e+01, 1.87114811799590e+02,
         9.90191814623914e+01, 1.80124575948747e+01)
        
    ax = abs(x)
    if ax > 0.5e0:
        if ax > 4.0e0:
            if ax >= 5.8e0:
                erf = 1.0e0 if x >= 0 else -1.0e0
            else:
                x2 = x*x
                t = 1.0e0/x2
                top = (((r[0]*t+r[1])*t+r[2])*t+r[3])*t + r[4]
                bot = (((s[0]*t+s[1])*t+s[2])*t+s[3])*t + 1.0e0
                erf = (c-top/ (x2*bot))/ax
                erf = 0.5e0 + (0.5e0-np.exp(-x2)*erf)
                if x < 0.0e0: erf = -erf

        else:
            top = ((((((p[0]*ax+p[1])*ax+p[2])*ax+p[3])*ax+p[4])*ax+p[5])*ax+p[6])*ax + p[7]
            bot = ((((((q[0]*ax+q[1])*ax+q[2])*ax+q[3])*ax+q[4])*ax+q[5])*ax+q[6])*ax + q[7]    
            erf = 0.5e0 + (0.5e0-np.exp(-x*x)*top/bot)
            if x < 0.0e0: erf = -erf
    else:
        t = x*x
        top = ((((a[0]*t+a[1])*t+a[2])*t+a[3])*t+a[4]) + 1.0e0
        bot = ((b[0]*t+b[1])*t+b[2])*t + 1.0e0
        erf = x* (top/bot)
    return erf
    

from time import perf_counter

s = perf_counter()
print(erf(6))
f = perf_counter()
print(f'math: {f-s}')

s = perf_counter()
print(erf2(6))
f = perf_counter()
print(f'mine: {f-s}')

s = perf_counter()
print(erf4(6))
f = perf_counter()
print(f'cons: {f-s}')

def mean_squared_error(y_real: Vector, y_pred: Vector) -> float:
    """Returns the mean squared error

    Args:
        y_real (Vector): Real data
        y_pred (Vector): Predicted data

    Returns:
        float: mean squared error
    """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    
    return np.mean((y_real - y_pred)**2)

def mean_absolute_error(y_real: Vector, y_pred: Vector) -> float:
    """Returns the mean absolute error

    Args:
        y_real (Vector): Real data
        y_pred (Vector): Predicted data

    Returns:
        float: mean absolute error
    """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    
    return np.mean(np.abs(y_real - y_pred))

def expectation(x: Vector, p: Vector = None) -> float:
    """Returns the expected value

    Args:
        x (Vector): Values
        p (Vector, optional): Probability. Defaults to None.

    Returns:
        float: Expectation
    """
    x = np.asarray(x)
    if p is None: return np.mean(x) 
    else: np.array(p)
    return np.sum(x*p)
    
def variance(x: Vector, p: Vector = None) -> float:
    """Return the dispersion of the values

    Args:
        x (Vector): Values
        p (Vector, optional): Probability. Defaults to None.

    Returns:
        float: Variance
    """
    x = np.asarray(x)
    if p is None: p = (1/len(x)) * np.ones([1,len(x)])
    else: np.array(p) 
    return np.sum((x - np.mean(x))**2) / len(x)
    #Second option
    #return expectation(x**2,p) - expectation(x,p)**2

def standard_deviation(x: Vector, p: Vector = None) -> float:
    """Returns the standard deviation

    Args:
        x (Vector): Values
        p (Vector, optional): Probability. Defaults to None.

    Returns:
        float: Standard deviation
    """
    return np.sqrt(variance(x,p))

def asimetry_coeficient(x: Vector, p: Vector = None) -> float:
    x = np.array(x)
    if p is None: p = (1/len(x)) * np.ones([1,len(x)])
    else: np.array(p)
    return expectation(x - expectation(x, p), p)**3 / standard_deviation(x, p)**3

def apunt_coeficient(x: Vector, p: Vector = None) -> float:
    x = np.array(x)
    if p is None: p = (1/len(x)) * np.ones([1,len(x)])
    else: np.array(p)
    return expectation(x - expectation(x, p), p)**4 / standard_deviation(x, p)**4



#---------------------- Distributions -----------------------#


class Distribution:
    """Main class for plotting distributions
    """

    def plot_PMF(self, show = True) -> None:
        """Plots the Probability mass function of a distribution

        Args:
            show (bool, optional): Shows the plot or keep editing it. Defaults to True.
        """
        dist_name = self.__class__.__name__
        if not hasattr(self, 'PMF'):
            print(f'{dist_name} distribution doesnt support PMF plotting')
            return None
        plt.title(f'Probability mass function\n{dist_name}')
        data = []
        for i in range(self.plot + 1):
            data.append(self.PMF(i))
        plt.plot(data)
        if show: plt.show()
    
    def plot_CDF(self, show = True) -> None:
        """Plots the Cumulative distribution function of a distribution

        Args:
            show (bool, optional): Shows the plot or keep editing it. Defaults to True.
        """
        dist_name = self.__class__.__name__
        if not hasattr(self, 'CDF'):
            print(f'{dist_name} distribution doesnt support CDF plotting')
            return None
        plt.title(f'Cumulative distribution function\n{dist_name}')
        data = []
        for i in range(self.plot + 1):
            data.append(self.CDF(i))
        plt.step(range(len(data)), data, where='post')
        if show: plt.show()
    
    def plot_PDF(self, show = True) -> None:
        """Plots the Probability density function of a distribution

        Args:
            show (bool, optional): Shows the plot or keep editing it. Defaults to True.
        """
        dist_name = self.__class__.__name__
        if not hasattr(self, 'PDF'):
            print(f'{dist_name} distribution doesnt support PDF plotting')
            return None
        plt.title(f'Probability density function\n{dist_name}')
        data = []
        iter = np.linspace(-self.plot + self.origin, self.plot + self.origin, 1000)
        for i in iter:
            data.append(self.PDF(i))
        plt.plot(iter, data)
        if show: plt.show()


#------------------- Discrete Distributions --------------------#

class Bernoulli(Distribution):
    """
    Bernoulli Distribution
    ======================

    Discrete probability distribution of a random variable
    which takes the value 1 with probability`p`and the
    value 0 with probability `q = 1-p`

    Parameters
    ----------
    p : float
        Probability of success
    
    Attributes
    ----------
    q : float
        `1-p`

    mean : float
        Expected value / mean 

    variance : float
        Measure of dispersion

    skewness : float
        Measure of the asymmetry

    kurtosis: float
        Measure of the "tailedness"

    Examples
    --------
    >>> from intelligen.stats import Bernoulli
    >>> B = Bernoulli(0.65)
    >>> B.PMF(0)
    0.35
    >>> B.skewness
    -0.6289709020331511

    """
    
    def __init__(self, p: float) -> None:
        self.p, self.q = p, 1-p
        self.mean = p
        self.variance = p * self.q
        self.skewness = (self.q - p) / np.sqrt(self.variance)
        self.kurtosis = (1 - 6*self.variance) / self.variance
    
    def __add__(self, distribution: Distribution) -> Distribution:
        if isinstance(distribution, self.__class__):
            if self.p == distribution.p:
                return Binomial(2, self.p)
            else: print('Probability must be the same')
        
        elif isinstance(distribution, Binomial):
            if self.p == distribution.p:
                return Binomial(distribution.n + 1, self.p)
            else: print('Probability must be the same')
        
        else: print('Distributions must be compatible')
    
    def __radd__(self, distribution: Distribution) -> Distribution:
        return self + distribution
    
    def __mul__(self, coef: int) -> Distribution:
        if isinstance(coef, int) : return Binomial(coef, self.p)
        else: print('The coeficient must be an integer')
    
    def __rmul__(self, coef: int) -> Distribution:
        return self * coef
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'
        


    def PMF(self, k: int) -> float:
        """Returns the probability mass function P(x=k)

        Args:
            k (float): Value

        Returns:
            float: Probability
        """
        if   k == 0: return self.q
        elif k == 1: return self.p
        else: print('R{0,1}')
    
    def CDF(self, k: float) -> float:
        """Returns the Cumulative distribution function P(x<=k)

        Args:
            k (float): Value

        Returns:
            float: Cumulative probability
        """
        if   k < 0: return 0
        elif k < 1: return self.q
        else: return 1


class Binomial(Distribution):
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
    
    p : float
        Probability of success

    Attributes
    ----------
    q : float
        `1-p`

    mean : float
        Expected value / mean 

    variance : float
        Measure of dispersion

    skewness : float
        Measure of the asymmetry

    kurtosis: float
        Measure of the "tailedness"

    Examples
    --------
    >>> from intelligen.stats import Binomial
    >>> B = Binomial(17, 0.65)
    >>> B.PMF(10)
    0.1684553555671748
    >>> B.kurtosis
    -0.09437621202327084
    """
    
    def __init__(self, n: int, p: float) -> None:
        self.n, self.p, self.q = n, p, 1-p
        self.plot = n
        self.mean = n * p
        self.variance = n * p * self.q
        self.skewness = (self.q - p) / np.sqrt(self.variance)
        self.kurtosis = (1 - 6*p*self.q) / self.variance
    
    def __add__(self, distribution: Distribution) -> Distribution:
        if isinstance(distribution, self.__class__):
            if self.p == distribution.p:
                return Binomial(self.n + distribution.n, self.p)
            else: print('Probability must be the same')
        
        else: print('Distributions must be compatible')

    def __mul__(self, coef: int) -> Distribution:
        if isinstance(coef, int) : return Binomial(self.n * coef, self.p)
        else: print('The coeficient must be an integer')
    
    def __rmul__(self, coef: int) -> Distribution:
        return self * coef
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n={self.n}, p={self.p})'

    def PMF(self, k: int) -> float:
        """Returns the probability mass function P(x=k)

        Args:
            k (float): Value

        Returns:
            float: Probability
        """
        return combination(self.n, k) * self.p**k * (1 - self.p)**(self.n - k) 

    def CDF(self, k: float) -> float:
        """Returns the Cumulative distribution function P(x<=k)

        Args:
            k (float): Value

        Returns:
            float: Cumulative probability
        """
        result = 0
        for i in range(int(np.floor(k))+1):
            result += combination(self.n, i) * self.p**i * (1 - self.p)**(self.n - i)
        return result

class Geometric(Distribution):
    """
    Geometric Distribution
    ======================

    Discrete probability distribution that models the number of
    Bernoulli trials `k`, needed to get one success, 
    each with success probability ``p`


    Parameters
    ----------
    p : float
        Probability of success

    Attributes
    ----------
    mean : float
        Expected value / mean 

    variance : float
        Measure of dispersion

    skewness : float
        Measure of the asymmetry

    kurtosis: float
        Measure of the "tailedness"

    Examples
    --------
    >>> from intelligen.stats import Geometric
    >>> G = Geometric(0.65)
    >>> G.CDF(3)
    0.957125
    >>> G.variance
    0.8284023668639052
    """
    def __init__(self, p: float, plot: int = None) -> None:
        self.p = p
        if plot is None: self.plot = int(10/p)
        self.mean = 1 / p
        self.variance = (1 - p) / p**2
        self.skewness = (2 - p) / np.sqrt(1 - p)
        self.kurtosis = 6 + p**2 / (1 - p)

    def __add__(self, distribution: Distribution) -> Distribution:
        if isinstance(distribution, self.__class__):
            if self.p == distribution.p:
                return NegativeBinomial(2, self.p)
            else: print('Probability must be the same')
        
        elif isinstance(distribution, NegativeBinomial):
            if self.p == distribution.p:
                return NegativeBinomial(distribution.r + 1, self.p)
            else: print('Probability must be the same')
        
        else: print('Distributions must be compatible')

    def __radd__(self, distribution: Distribution) -> Distribution:
        return self + distribution

    def __mul__(self, coef: int) -> Distribution:
        if isinstance(coef, int) : return NegativeBinomial(coef, self.p)
        else: print('The coeficient must be an integer')
    
    def __rmul__(self, coef: int) -> Distribution:
        return self * coef
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'

    def PMF(self, k: int or Vector) -> float:
        """Returns the probability mass function P(x=k)

        Args:
            k (int): Value

        Returns:
            float: Probability
        """
        if isinstance(k, int):
            return (1 - self.p)**(k - 1) * self.p
        return (1 - self.p)**(np.array(k) - 1) * self.p
    
    def CDF(self, k: float) -> float:
        """Returns the Cumulative distribution function P(x<=k)

        Args:
            k (float): Value

        Returns:
            float: Cumulative probability
        """
        return 1 - (1 - self.p)**np.floor(k)


class NegativeBinomial(Distribution):
    """
    Negative Binomial Distribution
    =====================

    Discrete probability distribution that models the number
    of successes in a sequence of independent and identically
    distributed Bernoulli trials before a specified
    (non-random) number of failures (denoted r) occurs

    Parameters
    ----------
    r: int
        Number of successes
    
    p : float
        Probability of success

    Attributes
    ----------
    q : float
        `1-p`

    mean : float
        Expected value / mean 

    variance : float
        Measure of dispersion

    skewness : float
        Measure of the asymmetry

    kurtosis: float
        Measure of the "tailedness"

    Examples
    --------
    >>> from intelligen.stats import Binomial
    >>> B = Binomial(17, 0.65)
    >>> B.PMF(10)
    0.1684553555671748
    >>> B.kurtosis
    -0.09437621202327084
    """
    def __init__(self, r: int, p: float, plot: int = None) -> None:
        """Negative binomial distribution

        Args:
            r (int): number of success
            p (float): probability of success [0, 1]
        """
        self.r, self.p = r, p
        if plot is None: self.plot = int((2*(r+4)/(1-self.p)))
        self.mean = p*r / (1 - p)
        self.variance = p*r / (1 - p)**2
        self.skewness = (1 + p) / np.sqrt(p*r)
        self.kurtosis = 6/r + (1 - p)**2 / (p*r)

    def __add__(self, distribution: Distribution) -> Distribution:
        if isinstance(distribution, self.__class__):
            if self.p == distribution.p:
                return NegativeBinomial(self.r + distribution.r, self.p)
            else: print('Probability must be the same')
        
        else: print('Distributions must be compatible')

    def __mul__(self, coef: int) -> Distribution:
        if isinstance(coef, int) : return NegativeBinomial(self.r * coef, self.p)
        else: print('The coeficient must be an integer')
    
    def __rmul__(self, coef: int) -> Distribution:
        return self * coef
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r}, p={self.p})'

    def PMF(self, k: int) -> float:
        """Returns the probability mass function P(x=k)

        Args:
            k (int): number of failures

        Returns:
            float: Probability
        """
        return combination(k + self.r - 1, k) * (1 - self.p)**self.r * self.p**k
    
    def CDF(self, k: float) -> float:
        """Returns the Cumulative distribution function P(x<=k)

        Args:
            k (float): Value

        Returns:
            float: Cumulative probability
        """
        Binomial(k + self.r, self.p).CDF(k)
class Hypergeometric(Distribution):
    
    def __init__(self, N: int, n: int, r: int) -> None:
        self.N, self.n, self.r, self.plot = N, n, r, r

    def PMF(self, k: int) -> float:
        """Returns the probability mass function P(x=k)

        Args:
            k (int): number of failures

        Returns:
            float: Probability
        """
        return combination(self.r, k) * combination(self.N - self.r, self.n - k) / combination(self.N, self.n)


class Poisson(Distribution):

    def __init__(self, l: float, plot: float = None) -> None:
        self.l = l
        if plot is None: self.plot = int(l*2) + 5
    
    def PMF(self, k: int) -> float:
        """Returns the probability mass function P(x=k)

        Args:
            k (int): number of failures

        Returns:
            float: Probability
        """
        return np.exp(-self.l) * self.l**k / factorial(k)

class Normal(Distribution):

    def __init__(self, mu: float = 0, s: float = 1) -> None:
        self.mu, self.s, self.plot, self.origin = mu, s, 3*(s+1), mu
    
    def PDF(self, k: float) -> float:
        return np.exp((-1/2) * ((k - self.mu)/self.s)**2) / (self.s * np.sqrt(2*np.pi))
    
    def CDF(self, x: float) -> float:
        """Returns the Cumulative distribution function P(x<=k)

        Args:
            k (float): Value

        Returns:
            float: Cumulative probability
        """
        return 1/2 * (1 + erf((x - self.mu)/(self.s*np.sqrt(2))))

""" N = Normal()
print(N.CDF(1)) """


