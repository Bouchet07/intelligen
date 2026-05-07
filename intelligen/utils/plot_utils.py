from intelligen.config import config
from intelligen.utils.types import PlotReturnType

__all__ = ["plot", "grid", "scatter", "set_labels", "plot_surface", "scatter_3d"]

def _detect_backend_from_obj(obj) -> str | None:
    """Detect the backend based on the module of the passed figure/axes object."""
    if obj is None:
        return None

    module_name = obj.__class__.__module__
    if module_name.startswith('matplotlib'):
        return 'matplotlib'
    elif module_name.startswith('plotly'):
        return 'plotly'
    elif module_name.startswith('bokeh'):
        return 'bokeh'

    return None

# --- Backend Formatting Functions ---

def _format_with_matplotlib(title=None, xlabel=None, ylabel=None, ax=None):
    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def _format_with_plotly(title=None, xlabel=None, ylabel=None, ax=None):
    if ax is not None:
        if title: ax.update_layout(title_text=title)
        if xlabel: ax.update_xaxes(title_text=xlabel)
        if ylabel: ax.update_yaxes(title_text=ylabel)
    return ax

def _format_with_bokeh(title=None, xlabel=None, ylabel=None, ax=None):
    if ax is not None:
        if title: ax.title.text = title
        if xlabel: ax.xaxis.axis_label = xlabel
        if ylabel: ax.yaxis.axis_label = ylabel
    return ax

# --- Backend Plotting Functions ---

def _plot_with_matplotlib(ax, *args, **kwargs):
    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax
    ax.plot(*args, **kwargs)
    return ax

def _plot_with_plotly(ax, *args, **kwargs):
    import plotly.graph_objects as go
    x = args[0] if len(args) > 0 else None
    y = args[1] if len(args) > 1 else None

    ax = go.Figure() if ax is None else ax
    ax.add_trace(go.Scatter(x=x, y=y, mode='lines', **kwargs))
    return ax

def _plot_with_bokeh(ax, *args, **kwargs):
    from bokeh.plotting import figure
    x = args[0] if len(args) > 0 else None
    y = args[1] if len(args) > 1 else None

    # In Bokeh, the 'ax' is a figure
    ax = figure() if ax is None else ax
    ax.line(x, y, **kwargs)
    return ax

# --- Backend Grid Functions ---

def _grid_with_matplotlib(visible: bool, ax, *args, **kwargs):
    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax
    ax.grid(visible, *args, **kwargs)
    return ax

def _grid_with_plotly(visible: bool, ax, *args, **kwargs):
    if ax is not None:
        ax.update_xaxes(showgrid=visible)
        ax.update_yaxes(showgrid=visible)
    return ax

def _grid_with_bokeh(visible: bool, ax, *args, **kwargs):
    if ax is not None:
        ax.xgrid.visible = visible
        ax.ygrid.visible = visible
    return ax

# --- Backend Scatter Functions ---

def _scatter_with_matplotlib(ax, *args, **kwargs):
    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax
    ax.scatter(*args, **kwargs)
    return ax

def _scatter_with_plotly(ax, *args, **kwargs):
    import plotly.graph_objects as go
    x = args[0] if len(args) > 0 else None
    y = args[1] if len(args) > 1 else None

    ax = go.Figure() if ax is None else ax
    ax.add_trace(go.Scatter(x=x, y=y, mode='markers', **kwargs))
    return ax

def _scatter_with_bokeh(ax, *args, **kwargs):
    from bokeh.plotting import figure
    x = args[0] if len(args) > 0 else None
    y = args[1] if len(args) > 1 else None

    # In Bokeh, the 'ax' is a figure
    ax = figure() if ax is None else ax
    ax.scatter(x, y, **kwargs)
    return ax

# --- Backend Surface (3D) Functions ---

def _surface_with_matplotlib(x, y, z, ax=None, **kwargs):
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, **kwargs)
    return ax

def _surface_with_plotly(x, y, z, ax=None, **kwargs):
    import plotly.graph_objects as go
    ax = go.Figure() if ax is None else ax

    # Safely extract the universal 'label' parameter
    label = kwargs.pop('label', None)
    if label is not None:
        kwargs['name'] = label
        kwargs['showscale'] = False # Optional: Hides the colorbar for regression planes

    ax.add_trace(go.Surface(x=x, y=y, z=z, **kwargs))
    return ax

def _surface_with_bokeh(x, y, z, fig=None, **kwargs):
    raise NotImplementedError("Bokeh does not natively support 3D surface plots.")

# --- Backend Scatter 3D Functions ---

def _scatter3d_with_matplotlib(x, y, z, ax=None, **kwargs):
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, **kwargs)
    return ax

def _scatter3d_with_plotly(x, y, z, ax=None, **kwargs):
    import plotly.graph_objects as go
    ax = go.Figure() if ax is None else ax

    # Safely extract the universal 'label' parameter
    label = kwargs.pop('label', None)
    if label is not None:
        kwargs['name'] = label

    ax.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', **kwargs))
    return ax

def _scatter3d_with_bokeh(x, y, z, fig=None, **kwargs):
    raise NotImplementedError("Bokeh does not natively support 3D scatter plots.")

# --- Public API ---

def set_labels(title: str = None, xlabel: str = None, ylabel: str = None, ax=None) -> PlotReturnType:
    """Set the title and axis labels for the current plot."""
    backend = _detect_backend_from_obj(ax) or config.plot_backend

    backends = {
        "matplotlib": _format_with_matplotlib,
        "plotly": _format_with_plotly,
        "bokeh": _format_with_bokeh,
    }

    if backend not in backends:
        raise ValueError(f"Unsupported backend '{backend}'.")

    return backends[backend](title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)

def plot(*args, ax=None, **kwargs) -> PlotReturnType:
    """Plot a line using the configured backend."""
    backend = _detect_backend_from_obj(ax) or config.plot_backend

    backends = {
        "matplotlib": _plot_with_matplotlib,
        "plotly": _plot_with_plotly,
        "bokeh": _plot_with_bokeh,
    }

    if backend not in backends:
        raise ValueError(f"Unsupported backend '{backend}'.")

    return backends[backend](ax, *args, **kwargs)

def grid(*args, visible=None, ax=None, **kwargs) -> PlotReturnType:
    """Toggle grid on the current plot."""
    backend = _detect_backend_from_obj(ax) or config.plot_backend

    backends = {
        "matplotlib": _grid_with_matplotlib,
        "plotly": _grid_with_plotly,
        "bokeh": _grid_with_bokeh,
    }

    if backend not in backends:
        raise ValueError(f"Unsupported backend '{backend}'.")

    visible = True if visible is None else visible
    return backends[backend](visible, ax, *args, **kwargs)

def scatter(*args, ax=None, **kwargs) -> PlotReturnType:
    """Plot a scatter using the configured backend."""
    backend = _detect_backend_from_obj(ax) or config.plot_backend

    backends = {
        "matplotlib": _scatter_with_matplotlib,
        "plotly": _scatter_with_plotly,
        "bokeh": _scatter_with_bokeh,
    }

    if backend not in backends:
        raise ValueError(f"Unsupported backend '{backend}'.")

    return backends[backend](ax, *args, **kwargs)

def plot_surface(x, y, z, ax=None, **kwargs) -> PlotReturnType:
    """Plot a 3D surface using the configured backend."""
    backend = _detect_backend_from_obj(ax) or config.plot_backend

    backends = {
        "matplotlib": _surface_with_matplotlib,
        "plotly": _surface_with_plotly,
        "bokeh": _surface_with_bokeh,
    }

    if backend not in backends:
        raise ValueError(f"Unsupported backend '{backend}'.")

    return backends[backend](x, y, z, ax=ax, **kwargs)

def scatter_3d(x, y, z, ax=None, **kwargs) -> PlotReturnType:
    """Plot a 3D scatter using the configured backend."""
    backend = _detect_backend_from_obj(ax) or config.plot_backend

    backends = {
        "matplotlib": _scatter3d_with_matplotlib,
        "plotly": _scatter3d_with_plotly,
        "bokeh": _scatter3d_with_bokeh,
    }

    if backend not in backends:
        raise ValueError(f"Unsupported backend '{backend}'.")

    return backends[backend](x, y, z, ax=ax, **kwargs)
