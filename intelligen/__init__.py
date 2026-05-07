# ruff: noqa: F401
"""Top-level package for intelligen."""

from importlib import metadata

# These imports make submodules available to the user,
# e.g., so they can access `intelligen.constants`. The `noqa`
# comment at the top of the file tells Ruff this is intentional.
from . import (
    constants,
    integrate,
    intelligen,
    interpolate,
    linear_model,
    linregress,
    metrics,
    numeric,
    signals,
    special,
    stats,
    utils,
)
from .config import config

__version__ = metadata.version("intelligen")

# Define the public API for when a user does `from intelligen import *`
__all__ = [
    "constants",
    "integrate",
    "intelligen",
    "interpolate",
    "linear_model",
    "linregress",
    "metrics",
    "numeric",
    "signals",
    "special",
    "stats",
    "utils",
    "__version__", # It's good practice to include __version__ in __all__
]


