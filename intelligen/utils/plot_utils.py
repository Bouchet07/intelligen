from intelligen.config import config

__all__ = ["plot", "grid"]

# --- Backend Plotting Functions ---

def _plot_with_matplotlib(ax, *args, **kwargs):
    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax
    ax.plot(*args, **kwargs)
    return ax

def _plot_with_plotly(fig, *args, **kwargs):
    import plotly.graph_objects as go
    x = args[0] if len(args) > 0 else None
    y = args[1] if len(args) > 1 else None

    fig = go.Figure() if fig is None else fig
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', **kwargs))
    return fig

def _plot_with_bokeh(fig, *args, **kwargs):
    from bokeh.plotting import figure
    x = args[0] if len(args) > 0 else None
    y = args[1] if len(args) > 1 else None

    # In Bokeh, the 'ax' is a figure
    fig = figure() if fig is None else fig
    fig.line(x, y, **kwargs)
    return fig

# --- Backend Grid Functions ---

def _grid_with_matplotlib(visible: bool, ax, *args, **kwargs):
    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax
    ax.grid(visible, *args, **kwargs)
    return ax

def _grid_with_plotly(visible: bool, fig, *args, **kwargs):
    if fig is not None:
        fig.update_xaxes(showgrid=visible)
        fig.update_yaxes(showgrid=visible)
    return fig

def _grid_with_bokeh(visible: bool, fig, *args, **kwargs):
    if fig is not None:
        fig.xgrid.visible = visible
        fig.ygrid.visible = visible
    return fig

# --- Public API ---

def plot(*args, ax=None, **kwargs):
    """Plot a line using the configured backend."""
    backend = config.plot_backend

    backends = {
        "matplotlib": _plot_with_matplotlib,
        "plotly": _plot_with_plotly,
        "bokeh": _plot_with_bokeh,
    }

    if backend not in backends:
        raise ValueError(f"Unsupported backend '{backend}'.")

    return backends[backend](ax, *args, **kwargs)

def grid(*args, visible=None, ax=None, **kwargs):
    """Toggle grid on the current plot."""
    backend = config.plot_backend

    backends = {
        "matplotlib": _grid_with_matplotlib,
        "plotly": _grid_with_plotly,
        "bokeh": _grid_with_bokeh,
    }

    if backend not in backends:
        raise ValueError(f"Unsupported backend '{backend}'.")

    visible = True if visible is None else visible
    return backends[backend](visible, ax, *args, **kwargs)
