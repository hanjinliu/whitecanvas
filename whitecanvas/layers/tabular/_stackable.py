from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Sequence,
    TypeVar,
    overload,
)

import numpy as np
from cmap import Color
from numpy.typing import NDArray

from whitecanvas import theme
from whitecanvas.backend import Backend
from whitecanvas.layers import _legend
from whitecanvas.layers import group as _lg
from whitecanvas.layers._deserialize import construct_layer
from whitecanvas.layers.tabular import _jitter, _shared
from whitecanvas.layers.tabular import _plans as _p
from whitecanvas.layers.tabular._df_compat import DataFrameWrapper
from whitecanvas.types import (
    ColormapType,
    ColorType,
    Hatch,
    LineStyle,
    Orientation,
    XYYData,
)
from whitecanvas.utils.collections import OrderedSet
from whitecanvas.utils.normalize import as_any_1d_array, as_color_array, parse_texts
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Self


_DF = TypeVar("_DF")


class AreaCollection(_lg.LayerCollection[_lg.Area]):
    """Collection of lines."""

    _ATTACH_TO_AXIS = True

    def __init__(
        self,
        *areas: _lg.Area,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        fill_alpha: float = 0.2,
    ):
        super().__init__(*areas, name=name)
        self._orient = Orientation.parse(orient)
        self._fill_alpha = fill_alpha

    @classmethod
    def from_arrays(
        cls,
        data: list[XYYData],
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        fill_alpha: float = 0.2,
        backend: str | Backend | None = None,
    ) -> Self:
        areas = list[_lg.Area]()
        ori = Orientation.parse(orient)
        for d in data:
            area = _lg.Area.from_arrays(d.x, d.ydiff, d.y0, orient=ori, backend=backend)
            area.fill_alpha = fill_alpha
            areas.append(area)
        return cls(areas, name=name, orient=ori)

    @property
    def fill_alpha(self) -> float:
        return self._fill_alpha

    @property
    def width(self) -> NDArray[np.float32]:
        return np.array([area.line.width for area in self], dtype=np.float32)

    @width.setter
    def width(self, width: float | Sequence[float]):
        if is_real_number(width):
            _width = [width] * len(self)
        else:
            _width = np.asarray(width, dtype=np.float32)
        if len(_width) != len(self):
            raise ValueError(
                f"width must be a float or a sequence of length {len(self)}"
            )
        for area, w in zip(self, _width):
            area.line.width = w

    @property
    def color(self) -> NDArray[np.float32]:
        return np.array([area.color for area in self], dtype=np.float32)

    @color.setter
    def color(self, color: str | Sequence[str]):
        col = as_color_array(color, len(self))
        for area, c in zip(self, col):
            area.color = c

    @property
    def style(self) -> list[LineStyle]:
        return np.array([area.line.style for area in self], dtype=object)

    @style.setter
    def style(self, style: str | Sequence[str]):
        styles = as_any_1d_array(style, len(self), dtype=object)
        for area, s in zip(self, styles):
            area.line.style = s

    @property
    def hatch(self) -> list[str]:
        return np.array([area.fill.face.hatch for area in self], dtype=object)

    @hatch.setter
    def hatch(self, hatch: str | Sequence[str]):
        hatches = as_any_1d_array(hatch, len(self), dtype=object)
        for area, h in zip(self, hatches):
            area.fill.face.hatch = h

    @property
    def orient(self) -> Orientation:
        """Orientation of the filling areas."""
        return self._orient

    def with_hover_texts(self, text: str | Iterable[Any]) -> Self:
        if isinstance(text, str):
            texts = [text] * len(self)
        else:
            texts = [str(t) for t in text]
            if len(texts) != len(self):
                raise ValueError("Length of texts must match the number of lines.")
        for area, txt in zip(self, texts):
            area.line.with_hover_text(txt)
            area.fill.with_hover_text(txt)
        return self

    def with_hover_template(
        self,
        template: str,
        extra: Any | None = None,
    ) -> Self:
        """Define hover template to the layer."""
        if self._backend_name in ("plotly", "bokeh"):  # conversion for HTML
            template = template.replace("\n", "<br>")
        params = parse_texts(template, len(self), extra)
        # set default format keys
        if "i" not in params:
            params["i"] = np.arange(len(self))
        texts = [
            template.format(**{k: v[i] for k, v in params.items()})
            for i in range(len(self))
        ]
        return self.with_hover_texts(texts)


class DFArea(_shared.DataFrameLayerWrapper[AreaCollection, _DF], Generic[_DF]):
    def __init__(
        self,
        base: AreaCollection,
        source: DataFrameWrapper[_DF],
        categories: list[tuple[Any, ...]],
        stackby: tuple[str, ...],
        splitby: tuple[str, ...],
        color_by: _p.ColorPlan,
        width_by: _p.WidthPlan,
        style_by: _p.StylePlan,
        hatch_by: _p.HatchPlan,
    ):
        super().__init__(base, source)
        self._categories = categories
        self._stackby = stackby
        self._splitby = splitby
        self._color_by = color_by
        self._width_by = width_by
        self._style_by = style_by
        self._hatch_by = hatch_by

    @classmethod
    def from_table(
        cls,
        source: DataFrameWrapper[_DF],
        data: list[XYYData],
        categories: list[tuple[Any, ...]],
        color: str | tuple[str, ...] | None = None,
        width: float = 1.0,
        style: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        orient: Orientation = Orientation.VERTICAL,
        fill_alpha: float = 0.2,
        stackby: str | tuple[str, ...] | None = None,
        name: str | None = None,
        backend: str | Backend | None = None,
    ):
        splitby = _shared.join_columns(color, style, hatch, source=source)
        base = AreaCollection.from_arrays(
            data, name=name, orient=orient, fill_alpha=fill_alpha, backend=backend
        )
        self = cls(
            base,
            source,
            categories,
            stackby,
            splitby,
            color_by=_p.ColorPlan.default(),
            width_by=_p.WidthPlan.default(),
            style_by=_p.StylePlan.default(),
            hatch_by=_p.HatchPlan.default(),
        )
        if color is not None:
            self.update_color(color)
        self.update_width(width)
        if style is not None:
            self.update_style(style)
        if hatch is not None:
            self.update_hatch(hatch)
        self.with_hover_template("\n".join(f"{k}: {{{k}!r}}" for k in self._splitby))
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: str | Backend | None = None) -> Self:
        """Create a layer from a dictionary."""
        base = d["base"]
        if isinstance(base, dict):
            base = construct_layer(base, backend=backend)
        return cls(
            base=base,
            source=d["source"],
            categories=d["categories"],
            stackby=tuple(d["stackby"]),
            splitby=tuple(d["splitby"]),
            color_by=_p.ColorPlan.from_dict_or_plan(d["color_by"]),
            width_by=_p.WidthPlan.from_dict_or_plan(d["width_by"]),
            style_by=_p.StylePlan.from_dict_or_plan(d["style_by"]),
            hatch_by=_p.HatchPlan.from_dict_or_plan(d["hatch_by"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "source": self._source,
            "base": self.base.to_dict(),
            "categories": self._categories,
            "stackby": self._stackby,
            "splitby": self._splitby,
            "color_by": self._color_by,
            "width_by": self._width_by,
            "style_by": self._style_by,
            "hatch_by": self._hatch_by,
        }

    @property
    def orient(self) -> Orientation:
        """Orientation of the filling areas."""
        return self.base.orient

    @classmethod
    def from_table_stacked(
        cls,
        df: DataFrameWrapper[_DF],
        x: str | _jitter.JitterBase,
        y: str | _jitter.JitterBase,
        stackby: str | tuple[str, ...] | None,
        *,
        color: str | tuple[str, ...] | None = None,
        width: float = 1.0,
        style: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        fill_alpha: float = 0.2,
        backend: str | Backend | None = None,
    ) -> DFArea[_DF]:
        splitby = _shared.join_columns(stackby, color, style, hatch, source=df)
        if stackby is None:
            stackby = splitby
        elif isinstance(stackby, str):
            stackby = (stackby,)
        nstackcol = len(stackby)
        if isinstance(x, _jitter.JitterBase):
            xj = x
        else:
            xj = _jitter.IdentityJitter(x)
        if isinstance(y, _jitter.JitterBase):
            yj = y
        else:
            yj = _jitter.IdentityJitter(y)
        # pre-calculate all the possible xs
        all_x = list(OrderedSet(xj.map(df)))
        x_to_i = {x: i for i, x in enumerate(all_x)}

        def _hash_rule(x: float) -> int:
            return int(round(x * 1000))

        stack_cat = [sl for sl, _ in df.group_by(splitby[nstackcol:])]
        ycumsum = {(_hash_rule(_x), *_s): 0.0 for _x in all_x for _s in stack_cat}
        data = list[XYYData]()
        categories = list[tuple[Any, ...]]()
        for sl, sub in df.group_by(splitby):
            this_h = np.zeros_like(all_x)
            for _x, _h in zip(xj.map(sub), yj.map(sub)):
                this_h[x_to_i[_x]] = _h
            bottom = []
            _stack_cat = sl[nstackcol:]
            for _x, _h in zip(all_x, this_h):
                _x_hash = _hash_rule(_x)
                dy = ycumsum[(_x_hash, *_stack_cat)]
                bottom.append(dy)
                ycumsum[(_x_hash, *_stack_cat)] += _h
            categories.append(sl)
            data.append(XYYData(all_x, bottom, this_h + bottom))
        return DFArea.from_table(
            df, data, categories=categories, color=color, hatch=hatch, width=width,
            stackby=stackby, name=name, orient=orient, fill_alpha=fill_alpha,
            backend=backend,
        )  # fmt: skip

    @overload
    def update_color(self, value: ColorType) -> Self: ...

    @overload
    def update_color(
        self,
        by: str | Iterable[str],
        palette: ColormapType | None = None,
    ) -> Self: ...

    def update_color(self, by, /, palette=None):
        """Update the color rule of the layer."""
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        self._base_layer.color = color_by.generate(self._categories, self._splitby)
        self._color_by = color_by
        return self

    def update_width(self, value: float) -> Self:
        """Update width of the lines."""
        self._base_layer.width = value
        return self

    def update_style(self, by: str | Iterable[str], styles=None) -> Self:
        """Update the styling rule of the layer."""
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot style by a column other than {self._splitby}")
            style_by = _p.StylePlan.new(cov.columns, values=styles)
        else:
            style_by = _p.StylePlan.from_const(LineStyle(cov.value))
        self._base_layer.style = style_by.generate(self._categories, self._splitby)
        self._style_by = style_by
        return self

    def update_hatch(self, by: str | Iterable[str], hatches=None) -> Self:
        """Update the hatch rule of the layer."""
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot hatch by a column other than {self._splitby}")
            hatch_by = _p.HatchPlan.new(cov.columns, values=hatches)
        else:
            hatch_by = _p.HatchPlan.from_const(cov.value)
        self._base_layer.hatch = hatch_by.generate(self._categories, self._splitby)
        self._hatch_by = hatch_by
        for area in self.base:
            area.fill.edge.width = 1.0
        return self

    def move(self, dx: float = 0.0, dy: float = 0.0, autoscale: bool = True) -> Self:
        """Add a constant shift to the layer."""
        for layer in self._base_layer:
            old_data = layer.data
            new_data = old_data[0] + dx, old_data[1] + dy
            layer.data = new_data
        if autoscale and (canvas := self._canvas_ref()):
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self

    def with_hover_template(self, template: str) -> Self:
        """Define hover template to the layer."""
        extra = {}
        for i, key in enumerate(self._splitby):
            extra[key] = [row[i] for row in self._categories]
        self.base.with_hover_template(template, extra=extra)
        return self

    def _prep_legend_info(
        self,
    ) -> tuple[
        list[tuple[str, ColorType]],
        list[tuple[str, LineStyle]],
        list[tuple[str, Hatch]],
    ]:
        df = _shared.list_to_df(self._categories, self._splitby)
        color_entries = self._color_by.to_entries(df)
        style_entries = self._style_by.to_entries(df)
        hatch_entries = self._hatch_by.to_entries(df)
        return color_entries, style_entries, hatch_entries

    def _as_legend_item(self) -> _legend.LegendItemCollection | _legend.LineLegendItem:
        colors, styles, hatches = self._prep_legend_info()
        ncolor = len(colors)
        nstyle = len(styles)
        nhatch = len(hatches)
        widths = self._base_layer.width

        color_default = theme.get_theme().foreground_color
        hatch_default = Hatch.SOLID
        style_default = LineStyle.SOLID
        if ncolor == 1:
            _, color_default = colors[0]
        if nstyle == 1:
            _, style_default = styles[0]
        if nhatch == 1:
            _, hatch_default = hatches[0]

        face_color_default = Color([*color_default.rgba[:3], self.base.fill_alpha])
        if ncolor == 1 and nstyle == 1 and nhatch == 1:
            face = _legend.FaceInfo(face_color_default, hatch_default)
            edge = _legend.EdgeInfo(color_default, widths[0], style_default)
            return _legend.BarLegendItem(face, edge)
        items = []
        if ncolor > 1:
            items.append((", ".join(self._color_by.by), _legend.TitleItem()))
            for (label, color), w in zip(colors, widths):
                fc = Color([*color.rgba[:3], self.base.fill_alpha])
                face = _legend.FaceInfo(fc, hatch_default)
                edge = _legend.EdgeInfo(color, w, style_default)
                item = _legend.BarLegendItem(face, edge)
                items.append((label, item))
        if nstyle > 1:
            items.append((", ".join(self._style_by.by), _legend.TitleItem()))
            for (label, style), w in zip(styles, widths):
                face = _legend.FaceInfo(face_color_default, hatch_default)
                edge = _legend.EdgeInfo(color_default, w, style)
                item = _legend.BarLegendItem(
                    face,
                    edge,
                )
                items.append((label, item))
        if nhatch > 1:
            items.append((", ".join(self._hatch_by.by), _legend.TitleItem()))
            for (label, hatch), w in zip(hatches, widths):
                face = _legend.FaceInfo(face_color_default, hatch)
                edge = _legend.EdgeInfo(color_default, w, style_default)
                item = _legend.BarLegendItem(face, edge)
                items.append((label, item))
        return _legend.LegendItemCollection(items)
