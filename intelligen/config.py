import importlib.util


class Config:
    """Configuration class for intelligen. This class holds global settings that can be accessed and modified by users of the library."""

    def __init__(self):
        self._supported_plot_backends: list[str] = ["matplotlib", "plotly", "bokeh"]
        self._supported_progress_bars: list[str] = ["tqdm", "alive_progress"]
        self._eps: float = 1e-10

        # Auto-detect defaults using a helper method
        self._plot_backend: str | None = self._detect_first_available(self._supported_plot_backends)
        self._progress_bar: str | None = self._detect_first_available(self._supported_progress_bars)

    @staticmethod
    def _detect_first_available(packages: list[str]) -> str | None:
        """Find the first installed package from a list without importing it."""
        for pkg in packages:
            if importlib.util.find_spec(pkg) is not None:
                return pkg
        return None

    @property
    def supported_plot_backends(self) -> list[str]:
        return self._supported_plot_backends

    @property
    def plot_backend(self) -> str | None:
        return self._plot_backend

    @plot_backend.setter
    def plot_backend(self, value: str) -> None:
        if value not in self._supported_plot_backends:
            raise ValueError(
                f"Unsupported backend '{value}'. "
                f"Supported backends are: {self._supported_plot_backends}"
            )

        if importlib.util.find_spec(value) is None:
            raise ImportError(f"Backend '{value}' is not installed. Please install it to use this backend.")

        self._plot_backend = value

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Epsilon must be a positive number.")
        self._eps = value

    @property
    def supported_progress_bars(self) -> list[str]:
        return self._supported_progress_bars

    @property
    def progress_bar(self) -> str | None:
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, value: str) -> None:
        if value not in self._supported_progress_bars:
            raise ValueError(
                f"Unsupported progress bar '{value}'. "
                f"Supported progress bars are: {self._supported_progress_bars}"
            )

        if importlib.util.find_spec(value) is None:
            raise ImportError(f"Progress bar '{value}' is not installed. Please install it to use this progress bar.")

        self._progress_bar = value

    def __repr__(self) -> str:
        """Return a custom string representation for the Config class."""
        display_attrs = {
            k.lstrip('_'): v
            for k, v in self.__dict__.items()
            if not k.startswith('_supported')
        }
        lines = [f"  {key} = {repr(val)}" for key, val in display_attrs.items()]
        return "Config(\n" + ",\n".join(lines) + "\n)"

# Instantiate a single instance to be used across the library
config = Config()
