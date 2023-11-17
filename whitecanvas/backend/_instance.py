_INSTALLED_MODULES = {}


class Backend:
    """The backend of plots."""

    _default: str = "matplotlib"

    def __init__(self, name: "Backend | str | None" = None) -> None:
        if name is None:
            name = self._default
        elif isinstance(name, Backend):
            name = name._name
        if name in _INSTALLED_MODULES:
            self._mod = _INSTALLED_MODULES[name]
        else:
            from importlib import import_module

            _INSTALLED_MODULES[name] = self._mod = import_module(f"whitecanvas.backend.{name}")
        self._name = name
        self.__class__._default = name

    def __repr__(self) -> str:
        return f"<Backend {self._name!r}>"

    @property
    def name(self) -> str:
        """Name of the backend."""
        return self._name

    def has(self, attr: str) -> bool:
        """Check if the current backend has an object."""
        return hasattr(self._mod, attr)

    def get(self, attr: str):
        """Get an object from the current backend."""
        out = getattr(self._mod, attr, None)
        if out is None:
            raise RuntimeError(f"Backend {self._name!r} does not have {attr!r}")
        return out
