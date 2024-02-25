from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypeVar

from whitecanvas.canvas.dataframe._base import BaseCatPlotter, CatIterator
from whitecanvas.layers import tabular as _lt
from whitecanvas.layers.tabular import _jitter, _shared
from whitecanvas.types import ColorType, Hatch, Orientation, Symbol

if TYPE_CHECKING:
    from whitecanvas.canvas._base import CanvasBase

    NStr = str | Sequence[str]

_C = TypeVar("_C", bound="CanvasBase")
_DF = TypeVar("_DF")


class StackedCatPlotter(BaseCatPlotter[_C, _DF]):
    def __init__(
        self,
        canvas: _C,
        df: _DF,
        offset: str | tuple[str, ...] | None,
        value: str | None,
        orient: Orientation,
        stackby: str | tuple[str, ...] | None = None,
        update_labels: bool = False,
    ):
        super().__init__(canvas, df)
        if isinstance(stackby, str):
            stackby = (stackby,)
        elif stackby is None:
            pass  # will be determined later
        else:
            if any(not isinstance(o, str) for o in stackby):
                raise TypeError(
                    "Category column(s) must be specified by a string or a sequence "
                    f"of strings, got {stackby!r}."
                )
            stackby = tuple(stackby)
        self._offset: tuple[str, ...] = offset
        self._cat_iter = CatIterator(self._df, offset)
        self._value = value
        self._stackby = stackby
        self._orient = orient
        self._update_labels = update_labels

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(offset={self._offset!r}, value={self._value!r}, "
            f"orient={self._orient!r}, stackby={self._stackby!r})"
        )

    def add_bars(
        self,
        *,
        color: ColorType | NStr | None = None,
        hatch: Hatch | NStr | None = None,
        name: str | None = None,
        extent: float = 0.8,
    ):
        canvas = self._canvas()
        df = self._df
        _splitby, _dodge = _shared.norm_dodge(
            df, self._offset, color, hatch, dodge=False
        )  # fmt: skip
        _pos_map = self._cat_iter.prep_position_map(_splitby, dodge=_dodge)

        xj = _jitter.CategoricalJitter(_splitby, _pos_map)
        yj = _jitter.IdentityJitter(self._value).check(df)

        _extent = self._cat_iter.zoom_factor(_dodge) * extent
        if not self._orient.is_vertical:
            xj, yj = yj, xj
        layer = _lt.DFBars.from_table_stacked(
            df, xj, yj, self._stackby, name=name, color=color, hatch=hatch,
            extent=_extent, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.update_color(color, palette=canvas._color_palette)
        elif color is None:
            layer.update_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def add_area(
        self,
        *,
        color: NStr | None = None,
        width: float = 2.0,
        hatch: NStr | None = None,
        style: NStr | None = None,
        name: str | None = None,
    ):
        canvas = self._canvas()
        df = self._df
        _splitby, _dodge = _shared.norm_dodge(
            df, self._offset, color, hatch, dodge=False
        )  # fmt: skip
        _pos_map = self._cat_iter.prep_position_map(_splitby, dodge=_dodge)

        xj = _jitter.CategoricalJitter(_splitby, _pos_map)
        yj = _jitter.IdentityJitter(self._value).check(df)
        layer = _lt.DFArea.from_table_stacked(
            self._df, xj, yj, self._stackby, name=name, color=color, hatch=hatch,
            style=style, width=width, orient=self._orient, backend=canvas._get_backend()
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.update_color(color, palette=canvas._color_palette)
        elif color is None:
            layer.update_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def add_stem(
        self,
        *,
        color: NStr | None = None,
        symbol: Symbol = Symbol.CIRCLE,
        size: float | None = None,
        width: float | None = None,
        style: NStr | None = None,
        name: str | None = None,
    ):
        ...
