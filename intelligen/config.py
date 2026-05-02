class Config:
    def __init__(self):
        self._supported_plot_backends: list[str] = ["matplotlib", "plotly", "bokeh"]
        self._plot_backend: str | None = None
        self._eps: float = 1e-10
        self._supported_progress_bars: list[str] = ["tqdm", "alive_progress"]
        self._progress_bar: str | None = None

        try:
            import matplotlib
            self._plot_backend = "matplotlib"
        except ImportError:
            try:
                import plotly
                self._plot_backend = "plotly"
            except ImportError:
                try:
                    import bokeh
                    self._plot_backend = "bokeh"
                except ImportError:
                    pass

        try:
            import tqdm
            self._progress_bar = "tqdm"
        except ImportError:
            try:
                import alive_progress
                self._progress_bar = "alive_progress"
            except ImportError:
                pass




    @property
    def supported_plot_backends(self):
        return self._supported_plot_backends

    @property
    def plot_backend(self):
        return self._plot_backend

    @plot_backend.setter
    def plot_backend(self, value):
        if value not in self._supported_plot_backends:
            raise ValueError(
                f"Unsupported backend '{value}'. "
                f"Supported backends are: {self._supported_plot_backends}"
            )
        try:
            exec(f"import {value}")
        except ImportError:
            raise ImportError(f"Backend '{value}' is not installed. Please install it to use this backend.")

        self._plot_backend = value

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, value):
        if value <= 0:
            raise ValueError("Epsilon must be a positive number.")
        self._eps = value

    @property
    def supported_progress_bars(self):
        return self._supported_progress_bars

    @property
    def progress_bar(self):
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, value):
        if value not in self._supported_progress_bars:
            raise ValueError(
                f"Unsupported progress bar '{value}'. "
                f"Supported progress bars are: {self._supported_progress_bars}"
            )
        try:
            exec(f"import {value}")
        except ImportError:
            raise ImportError(f"Progress bar '{value}' is not installed. Please install it to use this progress bar.")

        self._progress_bar = value

    def __repr__(self):
        display_attrs = {
            k.lstrip('_'): v
            for k, v in self.__dict__.items()
            if not k.startswith('_supported')
        }
        # Format them into a nice indented list
        lines = [f"  {key} = {repr(val)}" for key, val in display_attrs.items()]
        return "Config(\n" + ",\n".join(lines) + "\n)"

# Instantiate a single instance to be used across the library
config = Config()

