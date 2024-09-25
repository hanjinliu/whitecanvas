from __future__ import annotations

import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from cmap import Color, Colormap

from whitecanvas.canvas._palette import ColorPalette


@lru_cache(maxsize=1)
def catalog_names():
    return set(Colormap.catalog().keys())


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, dict):
            if isinstance(color := obj.get("color"), np.ndarray):
                if color.ndim == 1:
                    return Color(color).hex
                else:
                    return [Color(c).hex for c in color]
            else:
                return super().default(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, Color):
            return obj.hex
        elif isinstance(obj, Colormap):
            if obj.name in catalog_names():
                return obj.name
            return [(float(k), v.hex) for k, v in obj.color_stops]
        elif isinstance(obj, ColorPalette):
            return self.default(obj._cmap)
        elif hasattr(obj, "to_dict"):
            return color_to_hex(obj.to_dict())
        elif callable(obj):
            raise TypeError(f"Function {obj!r} is not JSON serializable")
        return super().default(obj)


def color_to_hex(d: dict[str, Any]) -> dict[str, Any]:
    to_update = {}
    for k, v in d.items():
        if isinstance(v, dict):
            to_update[k] = color_to_hex(v)
        elif isinstance(v, (tuple, list)):
            to_update[k] = [color_to_hex(e) if isinstance(e, dict) else e for e in v]
        elif k == "color" and isinstance(v, np.ndarray):
            if v.ndim == 1:
                to_update[k] = Color(v).hex
            else:
                to_update[k] = [Color(c).hex for c in v]
    d.update(to_update)
    return d
