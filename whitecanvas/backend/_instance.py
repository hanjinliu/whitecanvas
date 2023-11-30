from contextlib import contextmanager


_INSTALLED_MODULES = {}

_DEFAULT_APP = {
    "pyqtgraph": "qt",
    "vispy": "qt",
}


class Backend:
    """The backend of plots."""

    _default: str = "matplotlib"

    def __init__(self, name: "Backend | str | None" = None) -> None:
        if name is None:
            name = self._default
        elif isinstance(name, Backend):
            name = name._name
        if ":" in name:
            name, app = name.split(":", maxsplit=1)
            # normalize app name
            if app == "nb":
                app = "notebook"
        else:
            app = _DEFAULT_APP.get(name, "default")
        if name in _INSTALLED_MODULES:
            self._mod = _INSTALLED_MODULES[name]
        else:
            from importlib import import_module

            _INSTALLED_MODULES[name] = self._mod = import_module(
                f"whitecanvas.backend.{name}"
            )
        self._name: str = name
        self._app: str = app
        self.__class__._default = name

    def __repr__(self) -> str:
        return f"<Backend {self._name!r}>"

    @property
    def name(self) -> str:
        """Name of the backend."""
        return self._name

    @property
    def app(self) -> str:
        """Name of the application."""
        return self._app

    def has(self, attr: str) -> bool:
        """Check if the current backend has an object."""
        return hasattr(self._mod, attr)

    def get(self, attr: str):
        """Get an object from the current backend."""
        out = getattr(self._mod, attr, None)
        if out is None:
            raise RuntimeError(f"Backend {self._name!r} does not have {attr!r}")
        return out


class DummyObject:
    def __init__(self, *args, **kwargs):
        pass


class DummyBackendModule:
    def __init__(self):
        self.Canvas = DummyObject

    def __getattr__(self, key):
        raise RuntimeError(f"Cannot install {key!r} before finishing layouting.")


_dummy_backend_module = DummyBackendModule()


@contextmanager
def patch_dummy_backend():
    dummy_name = "."
    while dummy_name in _INSTALLED_MODULES:
        dummy_name += "."
    _INSTALLED_MODULES[dummy_name] = _dummy_backend_module
    try:
        yield dummy_name
    finally:
        del _INSTALLED_MODULES[dummy_name]
