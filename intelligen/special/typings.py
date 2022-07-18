from __future__ import annotations

from typing import List, Union, Callable, Tuple
from numbers import Real, Complex
from numpy.polynomial import Polynomial as cls_pol
import numpy as np

# from ..stats.ABCDistribution import ABCDistribution

_typing = ['List', 'Union', 'Callable', 'Tuple']

_numbers = ['Real', 'Complex']

_mine = ['Vector', 'Vector_int', 'Matrix2D',
         'Function', 'Function2d', 'Polynomial',
         'integer', 'vector']

__all__ = _typing + _numbers + _mine

# Type Alias

Vector = List[Real]
Vector_int = List[int]

Matrix2D = List[Vector]

Function = Callable[[Real], Real]
Function2d = Callable[[Real, Real], Real]

Polynomial = cls_pol
# Distribution = ABCDistribution

# Variables

integer = (int, np.integer)
vector = (list, tuple, np.ndarray)


#Post
# from ..stats.ABCDistribution import ABCDistribution
# Distribution = ABCDistribution
