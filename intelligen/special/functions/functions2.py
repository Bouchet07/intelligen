import numpy as np
#from scipy.special import comb, factorial

#from constants import golden, igolden
from ...constants import *

__all__ = ['factorial', 'comb',
           'fibonacci', 'binet']

def factorial(n: int) -> int:
    """
    Compute the factorial of a non-negative integer n.

    Parameters
    ----------
    n : int
        Non-negative integer to compute the factorial of.

    Returns
    -------
    int
        The factorial of n (n!).
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def comb(n: int, k: int, exact: bool = False) -> float:
    """
    Compute the binomial coefficient "n choose k".

    Parameters
    ----------
    n : int
        Total number of items.
    k : int
        Number of items to choose.
    exact : bool, optional
        If True, return an exact integer result. If False, return a floating-point result.
        Default is False.

    Returns
    -------
    float or int
        The binomial coefficient C(n, k).
    """
    if k < 0 or k > n:
        return 0
    if exact:
        return int(factorial(n) // (factorial(k) * factorial(n - k)))
    else:
        return factorial(n) / (factorial(k) * factorial(n - k))

def fibonacci(n: int, list: bool = False, start_points = None) -> int:
    """
    Fibonacci Numbers
    =================
    In mathematics, the Fibonacci numbers, commonly denoted Fn,
    form a sequence, the Fibonacci sequence, in which each number
    is the sum of the two preceding ones.

    Parameters
    ----------
    n : int
        nth element of the Fibonacci sequence
    list : bool, optional
        Shows the whole sequence until nth number, by default False
    start_points : Vector_int, optional
        Initial numbers of the sequence, by default [0, 1]

    Returns
    -------
    int
        nth element of the Fibonacci sequence
    """    """"""
    if start_points is None: start_points = [0, 1]
    f0, f1 = start_points

    if list:
        if n == 0: return [f0]
        if n == 1: return [f0, f1]

        F = start_points.copy()
        if n > 0:
            for i in range(1, n):
                F.append(F[i] + F[i-1])
        else:
            for _ in range(-n):
                F.insert(0, F[1] - F[0])

        return F
    else:
        if n == 0: return f0
        if n == 1: return f1

        if n > 0:
            for i in range(n-1):
                f1, f0 = f1 + f0, f1
            return f1
        else:
            for i in range(-n):
                f0, f1 = f1 - f0, f0
            return f0
            # Another way:
            #return (-1)**(n+1) * fibonacci(-n)

def binet(n: float):
    return (golden**n - igolden**n)/np.sqrt(5)

