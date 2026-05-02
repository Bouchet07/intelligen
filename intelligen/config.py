class Config:
    def __init__(self):
        self._supported_backends = ["matplotlib", "plotly"]
        self._plot_backend = "matplotlib"

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

