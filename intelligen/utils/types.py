from typing import TYPE_CHECKING, Any

__all__ = ["PlotReturnType"]

if TYPE_CHECKING:
    # Use standard try/except ImportError blocks. Type checkers understand these!
    try:
        from matplotlib.axes import Axes as MatplotlibAxes
    except ImportError:
        class MatplotlibAxes: pass  # Evaluates as an empty strict object, NOT Any

    try:
        from plotly.graph_objects import Figure as PlotlyFigure
    except ImportError:
        class PlotlyFigure: pass

    try:
        from bokeh.plotting import Figure as BokehFigure
    except ImportError:
        class BokehFigure: pass

    # Because there are no 'Any' types here, strict linting is preserved.
    PlotReturnType = MatplotlibAxes | PlotlyFigure | BokehFigure
else:
    PlotReturnType = Any



