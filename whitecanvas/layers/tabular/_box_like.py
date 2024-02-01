"""Layer with a dataframe bound to it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np
from cmap import Color

from whitecanvas import theme
from whitecanvas.backend import Backend
from whitecanvas.layers import _mixin
from whitecanvas.layers import group as _lg
from whitecanvas.layers.tabular import _plans as _p
from whitecanvas.layers.tabular import _shared
from whitecanvas.layers.tabular._df_compat import DataFrameWrapper
from whitecanvas.types import (
    ColorType,
    Hatch,
    LineStyle,
    Orientation,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas.dataframe._base import CatIterator

    _FE = _mixin.AbstractFaceEdgeMixin[_mixin.FaceNamespace, _mixin.EdgeNamespace]

_DF = TypeVar("_DF")


def _splitby_dodge(
    source: DataFrameWrapper[_DF],
    offset: str | tuple[str, ...],
    color: str | tuple[str, ...] | None = None,
    hatch: str | tuple[str, ...] | None = None,
    dodge: str | tuple[str, ...] | bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(offset, str):
        offset = (offset,)
    if isinstance(dodge, bool):
        if dodge:
            _all = _shared.join_columns(color, hatch, source=source)
            dodge = tuple(c for c in _all if c not in offset)
        else:
            dodge = ()
    elif isinstance(dodge, str):
        dodge = (dodge,)
    else:
        dodge = tuple(dodge)
    splitby = _shared.join_columns(offset, color, hatch, dodge, source=source)
    return splitby, dodge


def _norm_color_hatch(
    color,
    hatch,
    cat: CatIterator[_DF],
) -> tuple[_p.ColorPlan, _p.HatchPlan]:
    color_cov = _shared.ColumnOrValue(color, cat.df)
    if color_cov.is_column:
        color_by = _p.ColorPlan.from_palette(color_cov.columns)
    elif color_cov.value is not None:
        color_by = _p.ColorPlan.from_const(Color(color_cov.value))
    else:
        color_by = _p.ColorPlan.default()
    hatch_cov = _shared.ColumnOrValue(hatch, cat.df)
    if hatch_cov.is_column:
        hatch_by = _p.HatchPlan.new(hatch_cov.columns)
    elif hatch_cov.value is not None:
        hatch_by = _p.HatchPlan.from_const(Hatch(hatch_cov.value))
    else:
        hatch_by = _p.HatchPlan.default()
    return color_by, hatch_by


class _BoxLikeMixin:
    _source: DataFrameWrapper[_DF]

    def __init__(
        self,
        categories: list[tuple],
        splitby: tuple[str, ...],
        color_by: _p.ColorPlan,
        hatch_by: _p.HatchPlan,
    ):
        self._splitby = splitby
        self._categories = categories
        self._color_by = color_by
        self._hatch_by = hatch_by
        self._get_base().face.color = color_by.generate(self._categories, self._splitby)
        self._get_base().face.hatch = hatch_by.generate(self._categories, self._splitby)

    def _get_base(self) -> _FE:
        """Just for typing."""
        return self._base_layer

    def with_color_palette(self, palette) -> Self:
        if self._color_by.is_const():
            raise ValueError("Cannot redraw color for a constant color")
        color_by = _p.ColorPlan.from_palette(self._color_by.by, palette=palette)
        self._get_base().face.color = color_by.generate(self._categories, self._splitby)
        self._color_by = color_by
        return self

    def with_color(self, color: ColorType) -> Self:
        color_by = _p.ColorPlan.from_const(Color(color))
        self._get_base().face.color = color_by.generate(self._categories, self._splitby)
        self._color_by = color_by
        return self

    def with_hatch_palette(self, choices) -> Self:
        if self._hatch_by.is_const():
            raise ValueError("Cannot redraw hatch for a constant hatch")
        hatch_by = _p.HatchPlan.new(self._hatch_by.by, values=choices)
        self._get_base().face.hatch = hatch_by.generate(self._categories, self._splitby)
        self._hatch_by = hatch_by
        return self

    def with_hatch(self, hatch: str | Hatch) -> Self:
        hatch_by = _p.HatchPlan.from_const(Hatch(hatch))
        self._get_base().face.hatch = hatch_by.generate(self._categories, self._splitby)
        self._hatch_by = hatch_by
        return self

    def with_edge(
        self,
        *,
        color: ColorType | None = None,
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        alpha: float = 1.0,
    ) -> Self:
        self._get_base().with_edge(color=color, width=width, style=style, alpha=alpha)
        return self


class DFViolinPlot(
    _shared.DataFrameLayerWrapper[_lg.ViolinPlot, _DF],
    _BoxLikeMixin,
    Generic[_DF],
):
    def __init__(
        self,
        cat: CatIterator[_DF],
        value: str,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        dodge: str | tuple[str, ...] | bool | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        shape: str = "both",
        backend: str | Backend | None = None,
    ):
        _splitby, dodge = _splitby_dodge(cat.df, cat.offsets, color, hatch, dodge)
        x, arr, categories = cat.prep_arrays(_splitby, value, dodge=dodge)
        _extent = cat.zoom_factor(dodge=dodge) * extent
        color_by, hatch_by = _norm_color_hatch(color, hatch, cat)
        base = _lg.ViolinPlot.from_arrays(
            x, arr, name=name, orient=orient, shape=shape, extent=_extent,
            backend=backend,
        )  # fmt: skip
        super().__init__(base, cat.df)
        _BoxLikeMixin.__init__(self, categories, _splitby, color_by, hatch_by)

    @property
    def orient(self) -> Orientation:
        """Orientation of the violins."""
        return self._base_layer.orient

    def with_shift(
        self,
        shift: float = 0.0,
    ) -> Self:
        for layer in self._base_layer:
            _old = layer.data
            layer.set_data(edge_low=_old.y0 + shift, edge_high=_old.y1 + shift)
        if canvas := self._canvas_ref():
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self


class DFBoxPlot(
    _shared.DataFrameLayerWrapper[_lg.BoxPlot, _DF], _BoxLikeMixin, Generic[_DF]
):
    def __init__(
        self,
        cat: CatIterator[_DF],
        value: str,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        dodge: str | tuple[str, ...] | bool | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        capsize: float = 0.1,
        backend: str | Backend | None = None,
    ):
        _splitby, dodge = _splitby_dodge(cat.df, cat.offsets, color, hatch, dodge)
        x, arr, categories = cat.prep_arrays(_splitby, value, dodge=dodge)
        _extent = cat.zoom_factor(dodge=dodge) * extent
        _capsize = cat.zoom_factor(dodge=dodge) * capsize
        color_by, hatch_by = _norm_color_hatch(color, hatch, cat)
        base = _lg.BoxPlot.from_arrays(
            x, arr, name=name, orient=orient, capsize=_capsize, extent=_extent,
            backend=backend,
        )  # fmt: skip
        super().__init__(base, cat.df)
        _BoxLikeMixin.__init__(self, categories, _splitby, color_by, hatch_by)

    @property
    def orient(self) -> Orientation:
        """Orientation of the violins."""
        return self._base_layer.orient

    def with_shift(
        self,
        shift: float = 0.0,
    ) -> Self:
        self._base_layer.with_shift(shift)
        return self


class _EstimatorMixin(_BoxLikeMixin):
    orient: Orientation

    def est_by_mean(self) -> Self:
        """Set estimator to mean."""

        def est_func(x):
            return np.mean(x)

        return self._update_estimate(est_func)

    def est_by_median(self) -> Self:
        """Set estimator to median."""

        def est_func(x):
            return np.median(x)

        return self._update_estimate(est_func)

    def err_by_sd(self, scale: float = 1.0, *, ddof: int = 1) -> Self:
        """Set error to standard deviation."""

        def err_func(x):
            _mean = np.mean(x)
            _sd = np.std(x, ddof=ddof) * scale
            return _mean - _sd, _mean + _sd

        return self._update_error(err_func)

    def err_by_se(self, scale: float = 1.0, *, ddof: int = 1) -> Self:
        """Set error to standard error."""

        def err_func(x):
            _mean = np.mean(x)
            _er = np.std(x, ddof=ddof) / np.sqrt(len(x)) * scale
            return _mean - _er, _mean + _er

        return self._update_error(err_func)

    def err_by_quantile(self, low: float = 0.25, high: float | None = None) -> Self:
        """Set error to quantile."""
        if low < 0 or low > 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {low}")
        if high is None:
            high = 1 - low
        elif high < 0 or high > 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {high}")

        def err_func(x):
            _qnt = np.quantile(x, [low, high])
            return _qnt[0], _qnt[1]

        return self._update_error(err_func)

    def _update_estimate(self, est_func: Callable[[np.ndarray], float]) -> Self:
        arrays = self._get_arrays()
        est = [est_func(arr) for arr in arrays]
        self._set_estimation_values(est)
        return self

    def _update_error(
        self,
        err_func: Callable[[np.ndarray], tuple[float, float]],
    ) -> Self:
        arrays = self._get_arrays()
        err_low = []
        err_high = []
        for arr in arrays:
            low, high = err_func(arr)
            err_low.append(low)
            err_high.append(high)
        self._set_error_values(err_low, err_high)
        return self


class DFPointPlot(
    _shared.DataFrameLayerWrapper[_lg.LabeledPlot, _DF], _EstimatorMixin, Generic[_DF]
):
    def __init__(
        self,
        cat: CatIterator[_DF],
        value: str,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        dodge: str | tuple[str, ...] | bool | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        backend: str | Backend | None = None,
    ):
        _splitby, dodge = _splitby_dodge(cat.df, cat.offsets, color, hatch, dodge)
        x, arr, categories = cat.prep_arrays(_splitby, value, dodge=dodge)
        _capsize = cat.zoom_factor(dodge=dodge) * capsize
        color_by, hatch_by = _norm_color_hatch(color, hatch, cat)
        base = _lg.LabeledPlot.from_arrays(
            x, arr, name=name, orient=orient, capsize=_capsize, backend=backend,
        )  # fmt: skip
        self._arrays = arr
        super().__init__(base, cat.df)
        _BoxLikeMixin.__init__(self, categories, _splitby, color_by, hatch_by)
        base.with_edge(color=theme.get_theme().foreground_color)
        self._orient = orient

    @property
    def orient(self) -> Orientation:
        """Orientation of the violins."""
        return self._orient

    def with_shift(
        self,
        shift: float = 0.0,
    ) -> Self:
        base = self._base_layer
        data = base.data
        if self._orient.is_vertical:
            base.set_data(data.x + shift, data.y)
        else:
            base.set_data(data.x, data.y + shift)
        return self

    def _get_arrays(self) -> list[np.ndarray]:
        return self._arrays

    def _set_estimation_values(self, est):
        if self.orient.is_vertical:
            self._base_layer.set_data(ydata=est)
        else:
            self._base_layer.set_data(xdata=est)

    def _set_error_values(self, err_low, err_high):
        mdata = self._base_layer.data
        if self.orient.is_vertical:
            self._base_layer.yerr.set_data(mdata.x, err_low, err_high)
        else:
            self._base_layer.xerr.set_data(err_low, err_high, mdata.y)


class DFBarPlot(
    _shared.DataFrameLayerWrapper[_lg.LabeledBars, _DF], _BoxLikeMixin, Generic[_DF]
):
    def __init__(
        self,
        cat: CatIterator[_DF],
        value: str,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        dodge: str | tuple[str, ...] | bool | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ):
        _splitby, dodge = _splitby_dodge(cat.df, cat.offsets, color, hatch, dodge)
        x, arr, categories = cat.prep_arrays(_splitby, value, dodge=dodge)
        _extent = cat.zoom_factor(dodge=dodge) * extent
        _capsize = cat.zoom_factor(dodge=dodge) * capsize
        color_by, hatch_by = _norm_color_hatch(color, hatch, cat)
        base = _lg.LabeledBars.from_arrays(
            x, arr, name=name, orient=orient, capsize=_capsize, extent=_extent,
            backend=backend,
        )  # fmt: skip
        self._arrays = arr
        super().__init__(base, cat.df)
        _BoxLikeMixin.__init__(self, categories, _splitby, color_by, hatch_by)
        base.with_edge(color=theme.get_theme().foreground_color)
        self._orient = orient

    @property
    def orient(self) -> Orientation:
        return self._base_layer.bars.orient

    def _get_arrays(self) -> list[np.ndarray]:
        return self._arrays

    def _set_estimation_values(self, est):
        if self.orient.is_vertical:
            self._base_layer.set_data(ydata=est)
        else:
            self._base_layer.set_data(xdata=est)

    def _set_error_values(self, err_low, err_high):
        mdata = self._base_layer.data
        if self.orient.is_vertical:
            self._base_layer.yerr.set_data(mdata.x, err_low, err_high)
        else:
            self._base_layer.xerr.set_data(err_low, err_high, mdata.y)
