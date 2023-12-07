from __future__ import annotations

from typing import Literal, TypeVar, overload, Callable, Any, TYPE_CHECKING
from functools import update_wrapper
from numpy.typing import ArrayLike
from whitecanvas.canvas import Canvas, SingleCanvas
from whitecanvas import layers as _l
from whitecanvas.core import new_canvas
from whitecanvas.layers import _mixin
from whitecanvas.types import (
    LineStyle,
    Symbol,
    ColorType,
    Alignment,
    ColormapType,
    FacePattern,
    Orientation,
    ArrayLike1D,
    _Void,
)

from ._current import current_canvas, current_grid

if TYPE_CHECKING:
    from typing_extensions import Concatenate, ParamSpec

    _P = ParamSpec("_P")

_F = TypeVar("_F")


def _copy_method(f: _F) -> _F:
    """Search for the corresponding Canvas method and wrap."""
    return _copy_add_method(f, "")


def _copy_add_method(f: _F, pref: str = "add_") -> _F:
    """Search for the corresponding Canvas.add_* method and wrap."""
    fname = f"{pref}{f.__name__}"

    def _inner(*args, **kwargs):
        meth = getattr(current_canvas(), fname)
        return meth(*args, **kwargs)

    update_wrapper(_inner, f)
    _inner.__doc__ = getattr(Canvas, fname).__doc__
    return _inner


@overload
def line(
    ydata: ArrayLike1D, *, name: str | None = None, color: ColorType | None = None,
    width: float = 1.0, style: LineStyle | str = LineStyle.SOLID, alpha: float = 1.0,
    antialias: bool = True,
) -> _l.Line:  # fmt: skip
    ...


@overload
def line(
    xdata: ArrayLike1D, ydata: ArrayLike1D, *, name: str | None = None,
    color: ColorType | None = None, width: float = 1.0,
    style: LineStyle | str = LineStyle.SOLID, alpha: float = 1.0, antialias: bool = True,
) -> _l.Line:  # fmt: skip
    ...


@overload
def line(
    xdata: ArrayLike1D, ydata: Callable[[ArrayLike1D], ArrayLike1D], *,
    name: str | None = None, color: ColorType | None = None, width: float = 1.0,
    style: LineStyle | str = LineStyle.SOLID, alpha: float = 1.0, antialias: bool = True,
) -> _l.Line:  # fmt: skip
    ...


@_copy_add_method
def line(*args, **kwargs):
    ...


@overload
def markers(
    xdata: ArrayLike1D, ydata: ArrayLike1D, *,
    name: str | None = None, symbol: Symbol | str = Symbol.CIRCLE,
    size: float = 12, color: ColorType | None = None, alpha: float = 1.0,
    pattern: str | FacePattern = FacePattern.SOLID,
) -> _l.Markers[_mixin.ConstFace, _mixin.ConstEdge, float]:  # fmt: skip
    ...


@overload
def markers(
    ydata: ArrayLike1D, *,
    name: str | None = None, symbol: Symbol | str = Symbol.CIRCLE,
    size: float = 12, color: ColorType | None = None, alpha: float = 1.0,
    pattern: str | FacePattern = FacePattern.SOLID,
) -> _l.Markers[_mixin.ConstFace, _mixin.ConstEdge, float]:  # fmt: skip
    ...


@_copy_add_method
def markers(*args, **kwargs):
    ...


@overload
def bars(
    center: ArrayLike1D, height: ArrayLike1D, *, bottom: ArrayLike1D | None = None,
    name=None, orient: str | Orientation = Orientation.VERTICAL,
    extent: float = 0.8, color: ColorType | None = None,
    alpha: float = 1.0, pattern: str | FacePattern = FacePattern.SOLID,
) -> _l.Bars[_mixin.ConstFace, _mixin.ConstEdge]:  # fmt: skip
    ...


@overload
def bars(
    height: ArrayLike1D, *, bottom: ArrayLike1D | None = None,
    name=None, orient: str | Orientation = Orientation.VERTICAL,
    extent: float = 0.8, color: ColorType | None = None,
    alpha: float = 1.0, pattern: str | FacePattern = FacePattern.SOLID,
) -> _l.Bars[_mixin.ConstFace, _mixin.ConstEdge]:  # fmt: skip
    ...


@_copy_add_method
def bars(*args, **kwargs):
    ...


@_copy_add_method
def band(
    xdata: ArrayLike1D,
    ylow: ArrayLike1D,
    yhigh: ArrayLike1D,
    *,
    name: str | None = None,
    orient: str | Orientation = Orientation.VERTICAL,
    color: ColorType | None = None,
    alpha: float = 1.0,
    pattern: str | FacePattern = FacePattern.SOLID,
) -> _l.Band:
    ...


@_copy_add_method
def infline(
    pos: tuple[float, float] = (0, 0),
    angle: float = 0.0,
    *,
    name: str | None = None,
    color: ColorType | None = None,
    width: float = 1.0,
    style: LineStyle | str = LineStyle.SOLID,
    alpha: float = 1.0,
    antialias: bool = True,
) -> _l.InfLine:
    ...


@_copy_add_method
def infcurve(
    model: Callable[Concatenate[Any, _P], Any],
    *,
    bounds: tuple[float, float] = (-float("inf"), float("inf")),
    name: str | None = None,
    color: ColorType | None = None,
    width: float = 1.0,
    style: str | LineStyle = LineStyle.SOLID,
    antialias: bool = True,
) -> _l.InfCurve[_P]:
    ...


@_copy_add_method
def errorbars(
    xdata: ArrayLike1D,
    ylow: ArrayLike1D,
    yhigh: ArrayLike1D,
    *,
    name: str | None = None,
    orient: str | Orientation = Orientation.VERTICAL,
    color: ColorType = "blue",
    width: float = 1,
    style: LineStyle | str = LineStyle.SOLID,
    antialias: bool = False,
    capsize: float = 0.0,
) -> _l.Errorbars:
    ...


@_copy_add_method
def hist(
    data: ArrayLike1D,
    *,
    bins: int | ArrayLike1D = 10,
    range: tuple[float, float] | None = None,
    density: bool = False,
    name: str | None = None,
    orient: str | Orientation = Orientation.VERTICAL,
    color: ColorType | None = None,
    alpha: float = 1.0,
    pattern: str | FacePattern = FacePattern.SOLID,
) -> _l.Bars:
    ...


@_copy_add_method
def rug(
    events: ArrayLike1D,
    *,
    low: float = 0.0,
    high: float = 1.0,
    name: str | None = None,
    orient: str | Orientation = Orientation.VERTICAL,
    color: ColorType = "black",
    width: float = 1.0,
    style: LineStyle | str = LineStyle.SOLID,
    antialias: bool = True,
    alpha: float = 1.0,
) -> _l.Rug:
    ...


@_copy_add_method
def spans(*args, **kwargs) -> _l.Spans:
    ...


@_copy_add_method
def image(
    image: ArrayLike,
    *,
    cmap: ColormapType = "gray",
    clim: tuple[float | None, float | None] | None = None,
    flip_canvas: bool = True,
    lock_aspect: bool = True,
) -> _l.Image:
    ...


@_copy_add_method
def text(
    x: ArrayLike1D,
    y: ArrayLike1D,
    string: list[str],
    *,
    color: ColorType = "black",
    size: float = 12,
    rotation: float = 0.0,
    anchor: str | Alignment = Alignment.BOTTOM_LEFT,
    family: str | None = None,
) -> _l.Texts[_mixin.ConstFace, _mixin.ConstEdge, _mixin.ConstFont]:
    ...


@_copy_add_method
def kde(
    data: ArrayLike1D,
    *,
    bottom: float = 0.0,
    name: str | None = None,
    orient: str | Orientation = Orientation.VERTICAL,
    band_width: float | Literal["scott", "silverman"] = "scott",
    color: ColorType | None = None,
    alpha: float = 1.0,
    pattern: str | FacePattern = FacePattern.SOLID,
) -> _l.Band:
    ...


@_copy_method
def cat(
    data: Any,
    by: str | None = None,
    *,
    orient: str | Orientation = Orientation.VERTICAL,
    offsets: float | ArrayLike1D | None = None,
    palette: ColormapType | None = None,
    update_labels: bool = True,
):
    ...


def show(block: bool = False):
    """Show the current canvas."""
    current_grid().show(block=block)
