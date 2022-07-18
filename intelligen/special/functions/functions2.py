import numpy as np


from ..typings import *
#from constants import golden, igolden
from ...constants import *
from scipy.special import erf, erfc, erfi, erfinv, factorial, comb

__all__ = ['erf', 'erfc', 'erfi', 'erfinv', 'factorial', 'comb',
           'fibonacci', 'binet']

def fibonacci(n: int, list: bool = False, start_points: Vector_int = None) -> int:
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

