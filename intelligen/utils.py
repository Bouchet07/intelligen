from .config import config

__all__ = ["plot"]

def _plot_with_matplotlib(ax, *args, **kwargs):
    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax
    ax.plot(*args, **kwargs)
    return ax

def _plot_with_plotly(fig, *args, **kwargs):
    import plotly.graph_objects as go
    # Plotly usually takes x and y explicitly
    x = args[0] if len(args) > 0 else None
    y = args[1] if len(args) > 1 else None

    fig = go.Figure() if fig is None else fig
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', **kwargs))

    return fig

def plot(*args, ax=None, **kwargs):
    """Plot a line."""
    backend = config.plot_backend

    # The Dispatcher Map
    backends = {
        "matplotlib": _plot_with_matplotlib,
        "plotly": _plot_with_plotly,
    }

    if backend not in backends:
        raise ValueError(f"Unsupported backend '{backend}'.")

    # Call the specific function
    return backends[backend](ax, *args, **kwargs)

def _grid_with_matplotlib(visible: bool, ax, *args, **kwargs):
    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax
    ax.grid(visible, *args, **kwargs)
    return ax

def _grid_with_plotly(visible: bool, fig, *args, **kwargs):
    """Not implemented."""
    return fig


def grid(*args, visible=None, ax=None, **kwargs):
    """Toggle grid on the current plot."""
    backend = config.plot_backend

    backends = {
        "matplotlib": _grid_with_matplotlib,
        "plotly": _grid_with_plotly,
    }

    if backend not in backends:
        raise ValueError(f"Unsupported backend '{backend}'.")

    if visible is None:
        visible = True  # Default to showing the grid if not specified

    return backends[backend](visible, ax, *args, **kwargs)


