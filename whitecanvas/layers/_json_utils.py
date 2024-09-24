from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import numpy as np
from cmap import Color, Colormap


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
            if obj.name in Colormap.catalog():
                return obj.name
            cmap_dict = obj.as_dict()
            return super().default(
                [(float(k), Color(v).hex) for k, v in cmap_dict.items()]
            )
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)
