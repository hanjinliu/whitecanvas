_INSTALLED_MODULES = {}


class Backend:
    def __init__(self, name: str | None = None) -> None:
        if name is None:
            name = "pyqtgraph"
        if name in _INSTALLED_MODULES:
            self._mod = _INSTALLED_MODULES[name]
        else:
            from importlib import import_module
            
            _INSTALLED_MODULES[name] = self._mod = import_module(
                f"neoplot.backend.{name}"
            )
        self._name = name
    
    def __repr__(self) -> str:
        return f"<Backend {self._name!r}>"
    
    def get(self, attr: str):
        """Get an object from the current backend."""
        out = getattr(self._mod, attr, None)
        if out is None:
            raise RuntimeError(f"Backend {self._name!r} does not have {attr!r}")
        return out
