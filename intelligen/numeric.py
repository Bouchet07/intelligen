import numpy as np
from intelligen.special import comb as combination
from intelligen.special.typings import *
from intelligen.constants import golden, igolden

def derivative(f: Function ,a: float, order: int=1, method: str='central', h: float=0.01):
    '''
    Derivative
    ========
    Compute the difference formula for `f'(a)` with step size `h`.

    Parameters
    ----------
    f : Function
        Vectorized function of one variable, MUST accept arrays as input
    a : float
        Compute derivative at `x = a`
    method : string
        Difference formula: (order=1) 
            central: `f(a+h) - f(a-h))/2h`
            forward: `f(a+h) - f(a))/h`
            backward: `f(a) - f(a-h))/h`

            by default 'central'
        [Information about higher order](https://en.wikipedia.org/wiki/Finite_difference)

    h : float
        Step size in difference formula,
        by default 0.01
    
    order: int
        Order of the derivative

    Returns
    -------
    float
        `f'(a)`      
    '''
    i = np.arange(order+1)
    if method == 'central':
        return 1/h**order * np.sum((-1)**i * combination(order, i) * f(a + (order/2 - i)*h))

    elif method == 'forward':
        return  1/h**order * np.sum((-1)**(order - i) * combination(order, i) * f(a + i*h))

    elif method == 'backward':
        return 1/h**order * np.sum((-1)**i * combination(order, i) * f(a - i*h))

    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


#---------------Find Root---------------#


def newton(f: Function, x: float, tol: float, iter: bool = False) -> float:
    """
    Newton
    ======
    Newton method to find a root of a function

    Parameters
    ----------
    f : Function
        Function
    x : float
        Start point
    tol : float
        Error tolerance
    iter : bool, optional
        Shows the iterations needed to find the solution,
        by default False

    Returns
    -------
    float
        The root
    int, optional
        Number of iterations
    
    Examples
    --------
    >>> def f(x): return x**3 + 2*x**2 + 10*x - 20
    >>> newton(f, 1, 0.01, True)
    (1.3688081886175318, 3)


    """
    n = 0
    while abs(f(x)) > tol:
        x = x - f(x) / derivative(f, x, h=1e-5)
        n += 1

    if iter:
        return x, n
    return x

def bisection(f: Function, xi: float, xf: float, tol: float, iter: bool = False) -> float:
    """
    Bisection
    =========
    Bisection method to find a root of a function

    Parameters
    ----------
    f : Function
        Function
    xi : float
        First point
    xf : float
        Second point
    tol : float
        Error tolerance
    iter : bool, optional
        Shows the iterations needed to find the solution,
        by default False

    Returns
    -------
    float
        The root
    int, optional
        Number of iterations
    
    Examples
    --------
    >>> def f(x): return x**3 + 2*x**2 + 10*x - 20
    >>> bisection(f, 1, 2, 0.01, True)
    (1.369140625, 9)

    """
    if f(xi) * f(xf) < 0:
        xm, n = (xi + xf) / 2, 1

        while abs(f(xm)) > tol:
            if f(xi) * f(xm) < 0:
                xf = xm 
                n += 1
            
            elif f(xm) * f(xf) < 0:
                xi = xm 
                n += 1
            
            xm = (xi + xf) / 2
            
        if iter:
            return xm, n
        return xm

    else:
        print("Invalid input")

def regula_falsi(f: Function, xi: float, xf: float, tol: float, iter: bool = False) -> float:
    """
    Regula Falsi
    ============
    Regula falsi method to find a root of a function
    

    Parameters
    ----------
    f : Function
        Function
    xi : float
        First point
    xf : float
        Second point
    tol : float
        Error tolerance
    iter : bool, optional
        Shows the iterations needed to find the solution,
        by default False

    Returns
    -------
    float
        The root
    int, optional
        Number of iterations
    
    Examples
    --------
    >>> def f(x): return x**3 + 2*x**2 + 10*x - 20
    >>> regula_falsi(f, 1, 2, 0.01, True)
    (1.3685009755999702, 4)

    """
    if f(xi) * f(xf) < 0:
        xm, n = (xi * f(xf) - xf * f(xi)) / (f(xf) - f(xi)), 1

        while abs(f(xm)) > tol:
            if f(xi) * f(xm) < 0:
                xf = xm
                n += 1

            if f(xm) * f(xf) < 0:
                xi = xm
                n += 1
            
            xm = (xi * f(xf) - xf * f(xi)) / (f(xf) - f(xi))

        if iter:
            return xm, n
        return xm
    else:
        print("Invalid input")


def secant(f: Function, x0: float, x1: float, tol: float, iter: bool = False) -> float:
    """
    Secant
    ======

    Secant method to find a root of a function

    Parameters
    ----------
    f : Function
        Function
    x0 : float
        First point
    x1 : float
        Second point
    tol : float
        Error tolerance
    iter : bool, optional
        Shows the iterations needed to find the solution,
        by default False

    Returns
    -------
    float
        The root
    int, optional
        Number of iterations
    
    Examples
    --------
    >>> def f(x): return x**3 + 2*x**2 + 10*x - 20
    >>> secant(f, 1, 2, 0.01, True)
    (1.369013325992566, 3)

    """
    x2, n = x1 - (f(x1) * (x1 - x0)) / (f(x1) - f(x0)), 1

    while abs(f(x2)) > tol:
        x0, x1 = x1, x2
        x2 = x1 - (f(x1) * (x1 - x0)) / (f(x1) - f(x0))
        n += 1
        
    if iter:
        return x2, n
    return x2



""" def fixed_point(g, dg, x, tol, iter = False):
  ''' f(x) = 0 ==> x = g(x) '''
  n = 1
  if abs(dg(x))<1:
    xa = x
    x = g(x)
    
    while abs(x-xa)>tol:
      xa = x
      x = g(x)
      n = n+1
    if iter: return x, n
    return x
  else:
    print("Doesn't converge")  """


def newton2(f: Function, x: float, tol: float, iter: bool = False) -> float:
    """
    Newton 2nd order
    ================

    Newton second order method to find a root of a function

    Parameters
    ----------
    f : Function
        Function
    x : float
        Start point
    tol : float
        Error tolerance
    iter : bool, optional
        Shows the iterations needed to find the solution,
        by default False

    Returns
    -------
    float
        The root
    int, optional
        Number of iterations
    
    Examples
    --------
    >>> def f(x): return x**3 + 2*x**2 + 10*x - 20
    >>> newton2(f, 1, 0.01, True)
    (1.3688081071467233, 2)

    """
    n = 0

    while abs(f(x)) > tol:
        fx, dfx, ddfx = f(x), derivative(f, x), derivative(f, x, 2)

        x1 = x - dfx / ddfx + np.sqrt(dfx**2 - 2*ddfx * fx) / ddfx
        x2 = x - dfx / ddfx - np.sqrt(dfx**2 - 2*ddfx * fx) / ddfx

        if abs(f(x1)) < abs(f(x2)):
            x = x1
        else:
            x = x2
        
        n += 1
    
    if iter:
        return x, n
    return x
