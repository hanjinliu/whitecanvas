from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from whitecanvas.backend import Backend
    from whitecanvas.layers._base import Layer

_TYPE_MAP: dict[str, Layer] = {}


def register_layer_type(cls: type[Layer], name: str | None = None) -> None:
    """Register a layer type."""
    if name is None:
        name = cls.__name__
    if name in _TYPE_MAP:
        warnings.warn(
            f"Layer type {name} is already registered.", RuntimeWarning, stacklevel=3
        )
    _TYPE_MAP[name] = cls
    return None


def construct_layer(d: dict[str, Any], backend: Backend | str | None = None) -> Layer:
    """Construct a layer from a dictionary."""
    if "type" not in d:
        raise ValueError(f"Layer dict must have a 'type' key, got {d!r}.")
    return _TYPE_MAP[d["type"]].from_dict(d, backend=backend)


def construct_layers(
    d: list[dict[str, Any]],
    backend: Backend | str | None = None,
) -> list[Layer]:
    """Construct a list of layers from a list of dictionaries."""
    return [construct_layer(c, backend=backend) for c in d]
