"""Top-level package for intelligen."""

from importlib import metadata

__version__ = metadata.version("intelligen")

submodules = [
        'AI',
        'constants',
        'integrate',
        'intelligen',
        'interpolate',
        'linear_model',
        'linregress',
        'metrics',
        'numeric',
        'signals',
        'stats',
        'special'
    ]

__all__ = submodules

from . import AI
from . import constants
from . import integrate
from . import intelligen
from . import interpolate
from . import linear_model
from . import linregress
from . import metrics
from . import numeric
from . import signals
from . import stats
from . import special
