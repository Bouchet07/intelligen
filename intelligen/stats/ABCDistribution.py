import abc
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

from ..special.typings import Real, Vector, Union, Matrix2D


class ABCDistribution(abc.ABC):
    '''
    A probability distribution is the mathematical function that gives the probabilities
    of occurrence of different possible outcomes for an experiment.

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
    '''
    @property
    @abc.abstractmethod
    def mean(self):
        """
        the expected value (also called expectation, mean, average, or first moment)
        is a generalization of the weighted average.
        """
        pass

    @property
    @abc.abstractmethod
    def variance(self):
        """
        variance is the expectation of the squared deviation of a random variable
        from its population mean or sample mean. Variance is a measure of dispersion
        """
        pass

    @property
    @abc.abstractmethod
    def skewness(self):
        """
        skewness is a measure of the asymmetry of the probability distribution
        of a real-valued random variable about its mean
        """
        pass

    @property
    @abc.abstractmethod
    def kurtosis(self):
        """
        kurtosis is a measure of the "tailedness" of the probability distribution
        of a real-valued random variable
        """
        pass
    
    @abc.abstractmethod
    def cdf(k: Union[Real,Vector]) -> Union[Real,Vector]:
        """
        Cumulative Distribution Function
        ================================
        Returns the Cumulative distribution function P(X<=k)

        Parameters
        ----------
        k : Real / Vector
            Value
        
        Returns
        -------
        Real / Vector:
            Cumulative probability
        """
        pass

    
    # @abc.abstractmethod
    def plot_cdf(self, ax = None, limit=None, **kwargs) -> plt.axes:
        """Plots the Cumulative distribution function of a distribution
        
        Returns
        -------
        plt.axes
            returns the axes of the plot
        """
        if ax is None: ax = plt.gca()
        if limit is None: limit = [0.01, 0.99]
        a, b = self.qf(limit)
        x = np.linspace(a, b, 100)
        ax.plot(x, self.cdf(x), **kwargs)
        ax.grid(True)
        return ax
        

Distribution = ABCDistribution

class ABCDiscreteDistribution(ABCDistribution):
    
    @abc.abstractmethod
    def pmf(self, k):
        """
        Probability Mass Function
        =========================
        Returns the probability mass function P(X=k)

        Parameters
        ----------
        k : Real / Vector
            Value
        
        Returns
        -------
        Real / Vector:
            Probability
        """
        pass
    
    @abc.abstractmethod
    def plot_pmf(self) -> plt.axes:
        """Plots the probability mass function

        Returns
        -------
        plt.axes
            returns the axes of the plot
        """
        pass

    @abc.abstractmethod
    def pdf(self, a, b):
        """
        
        Probability density function
        ============================
        Returns the Probability density function P(a<=X<=b)

        Parameters
        ----------
        a : Real / Vector
            Lower bound
        b : Real / Vector
            Upper bound

        Returns
        -------
        Real / Vector / Matrix:
            Density probability
        """
        pass
    
    # @abc.abstractmethod
    def plot_pdf(self, ax = None, limit=0.01) -> plt.axes:
        """Plots the Probability density function of a distribution
        
        Returns
        -------
        plt.axes
            returns the axes of the plot
        """
        pass
    
    @abc.abstractmethod
    def mgf(self, t: Union[Real,Vector]) -> Union[Real,Vector]:
        """
        Moment-generating function
        ==========================

        Parameters
        ----------
        t : Union[Real,Vector]
            _description_

        Returns
        -------
        Union[Real,Vector]
            _description_
        """
        pass

    @abc.abstractmethod    
    def cf(self, t: Union[Real,Vector]) -> Union[Real,Vector]:
        """
        Characteristic function
        =======================

        Parameters
        ----------
        t : Union[Real,Vector]
            _description_

        Returns
        -------
        Union[Real,Vector]
            _description_
        """
        pass



class ABCContinousDistribution(ABCDistribution):
    
    @abc.abstractmethod
    def pdf(self, x):
        """
        
        Probability density function
        ============================
        Returns the Probability density function P(a<=X<=b)

        Parameters
        ----------
        x: Array-like

        Returns
        -------
        Array-like:
            Density probability
        """
        pass

    # @abc.abstractmethod
    def plot_pdf(self, ax = None, limit=0.01, **kwargs) -> plt.axes:
        """Plots the Probability density function of a distribution
        
        Returns
        -------
        plt.axes
            returns the axes of the plot
        """
        if ax is None: ax = plt.gca()
        a = op.fsolve(lambda x:self.pdf(x)-limit, self.mean-np.sqrt(self.variance))[0]
        b = op.fsolve(lambda x:self.pdf(x)-limit, self.mean+np.sqrt(self.variance))[0]
        x = np.linspace(a,b,100)
        ax.plot(x, self.pdf(x), **kwargs)
        ax.grid(True)
        return ax
        
Distribution = ABCDistribution