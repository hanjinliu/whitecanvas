from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from whitecanvas.backend import Backend
    from whitecanvas.layers._base import Layer

_CLASS_MAP: dict[str, type[Layer]] = {}


def _pick_layer_class(name: str) -> type[Layer]:
    import importlib

    if name not in _CLASS_MAP:
        mod_name, cls_name = name.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        _CLASS_MAP[name] = getattr(mod, cls_name)
    return _CLASS_MAP[name]


def construct_layer(d: dict[str, Any], backend: Backend | str | None = None) -> Layer:
    """Construct a layer from a dictionary."""
    if "type" not in d:
        raise ValueError(f"Layer dict must have a 'type' key, got {d!r}.")
    typ = d["type"]
    if not isinstance(typ, str):
        raise ValueError(f"Layer type must be a string, got {typ!r}.")
    return _pick_layer_class(typ).from_dict(d, backend=backend)


def construct_layers(
    d: list[dict[str, Any]],
    backend: Backend | str | None = None,
) -> list[Layer]:
    """Construct a list of layers from a list of dictionaries."""
    return [construct_layer(c, backend=backend) for c in d]
