class Config:
    def __init__(self):
        self._supported_backends = ["matplotlib", "plotly"]
        self._plot_backend = "matplotlib"
        self._eps = 1e-10

    @property
    def supported_backends(self):
        return self._supported_backends

    @property
    def plot_backend(self):
        return self._plot_backend

    @plot_backend.setter
    def plot_backend(self, value):
        if value not in self._supported_backends:
            raise ValueError(
                f"Unsupported backend '{value}'. "
                f"Supported backends are: {self._supported_backends}"
            )
        self._plot_backend = value

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, value):
        if value <= 0:
            raise ValueError("Epsilon must be a positive number.")
        self._eps = value

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

