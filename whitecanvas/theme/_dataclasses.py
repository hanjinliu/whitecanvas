from __future__ import annotations

from dataclasses import Field, dataclass, field, fields
from enum import Enum

from cmap import Color, Colormap

from whitecanvas.types import Hatch, LineStyle, Symbol


@dataclass
class _BaseModel:
    mutable = True

    def __post_init__(self):
        self._set_mutable(False)

    def __setattr__(self, name, value):
        if not self.mutable:
            raise TypeError(f"{self.__class__.__name__} is immutable.")
        fld: Field = self.__class__.__dataclass_fields__[name]
        if (validator := fld.metadata.get("validator", None)) is not None:
            value = validator(value)
        super().__setattr__(name, value)

    def _set_mutable(self, value: bool = True):
        super().__setattr__("mutable", value)
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, _BaseModel):
                val._set_mutable(value)

    def copy(self):
        d = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, _BaseModel):
                val = val.copy()
            elif isinstance(val, (bool, int, float, str, tuple, Enum)):
                pass
            elif isinstance(val, (list, dict, set)):
                val = val.copy()
            else:
                val = type(val)(val)
            d[f.name] = val
        return self.__class__(**d)

    @classmethod
    def _validate_model(cls, value):
        if isinstance(value, cls):
            return value
        elif isinstance(value, dict):
            return cls(**value)
        else:
            raise TypeError(f"Cannot convert {type(value)} to {cls.__name__}.")


def _field(obj, validator=None):
    if isinstance(obj, type):
        if issubclass(obj, _BaseModel) and validator is None:
            validator = obj._validate_model
        return field(default_factory=obj, metadata={"validator": validator})
    else:
        if isinstance(obj, (int, str, float, bool, Enum, tuple, Color)):
            if validator is None:
                validator = type(obj)
            return field(default=obj, metadata={"validator": validator})
        else:
            return field(default_factory=lambda: obj, metadata={"validator": validator})


@dataclass
class Font(_BaseModel):
    """Font of texts."""

    family: str = _field("Arial")
    size: int = _field(11)
    color: Color = _field(Color("black"))


@dataclass
class Line(_BaseModel):
    """Line style."""

    width: float = _field(2.0)
    style: LineStyle = _field(LineStyle.SOLID)


@dataclass
class Markers(_BaseModel):
    """Markers of points."""

    size: float = _field(8.0)
    hatch: Hatch = _field(Hatch.SOLID)
    symbol: Symbol = _field(Symbol.CIRCLE)


@dataclass
class Bars(_BaseModel):
    """Bar style."""

    extent: float = _field(0.8)
    hatch: Hatch = _field(Hatch.SOLID)


@dataclass
class ErrorBars(_BaseModel):
    """Error bar style."""

    width: float = _field(2.0)
    style: LineStyle = _field(LineStyle.SOLID)


def _validate_canvas_size(size) -> tuple[float, float]:
    w, h = size
    w = int(w)
    h = int(h)
    if w <= 0 or h <= 0:
        raise ValueError("Canvas size must be positive.")
    return w, h


@dataclass
class Theme(_BaseModel):
    """Plot theme."""

    font: Font = _field(Font)
    line: Line = _field(Line)
    markers: Markers = _field(Markers)
    bars: Bars = _field(Bars)
    errorbars: ErrorBars = _field(ErrorBars)
    foreground_color: Color = _field(Color("black"))
    background_color: Color = _field(Color("white"))
    canvas_size: tuple[float, float] = _field((800, 600), _validate_canvas_size)
    palette: Colormap = _field(Colormap("tab10"), Colormap)


LIGHT_THEME = Theme()
DARK_THEME = Theme(
    font=Font(color="white"),
    foreground_color="#FFFFFF",
    background_color="#000000",
    palette="tab10_light",
)
