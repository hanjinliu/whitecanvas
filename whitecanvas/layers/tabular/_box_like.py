"""Layer with a dataframe bound to it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, Iterable, TypeVar

import numpy as np
from cmap import Color

from whitecanvas import theme
from whitecanvas.backend import Backend
from whitecanvas.layers import _mixin
from whitecanvas.layers import group as _lg
from whitecanvas.layers.tabular import _plans as _p
from whitecanvas.layers.tabular import _shared
from whitecanvas.layers.tabular._df_compat import DataFrameWrapper, parse
from whitecanvas.types import (
    ColorType,
    Hatch,
    LineStyle,
    Orientation,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    _FE = _mixin.AbstractFaceEdgeMixin[_mixin.FaceNamespace, _mixin.EdgeNamespace]

_DF = TypeVar("_DF")


class _BoxLikeMixin:
    _source: DataFrameWrapper[_DF]

    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        offset: str | tuple[str, ...],
        value: str,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
    ):
        if isinstance(offset, str):
            offset = (offset,)
        splitby = _shared.join_columns(offset, color, hatch, source=source)
        self._y = value
        self._splitby = splitby
        self._color_by = _p.ColorPlan.default()
        self._hatch_by = _p.HatchPlan.default()
        self._offset_by = _p.OffsetPlan.default().more_by(*offset)
        self._source = source

    @property
    def color(self) -> _p.ColorPlan | _p.ColormapPlan:
        """Return the object describing how the plot is colored."""
        return self._color_by

    @property
    def hatch(self) -> _p.HatchPlan:
        """Return the object describing how the plot is hatched."""
        return self._hatch_by

    def _get_base(self) -> _FE:
        """Just for typing."""
        return self._base_layer

    def with_color(self, by: str | Iterable[str], palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            other_by = _shared.unique_tuple(self._offset_by.by, self._hatch_by.by)
            by_all = _shared.unique_tuple(cov.columns, other_by)
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
            self._splitby = by_all
            _, self._labels = self._generate_datasets()
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        self._get_base().face.color = color_by.generate(self._labels, self._splitby)
        self._color_by = color_by
        return self

    def with_hatch(
        self,
        by: str | Iterable[str],
        choices=None,
    ) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            other_by = _shared.unique_tuple(self._offset_by.by, self._color_by.by)
            by_all = _shared.unique_tuple(other_by, cov.columns)
            hatch_by = _p.HatchPlan.new(cov.columns, values=choices)
            self._splitby = by_all
            _, self._labels = self._generate_datasets()
        else:
            hatch_by = _p.HatchPlan.from_const(Hatch(cov.value))
        self._get_base().face.hatch = hatch_by.generate(self._labels, self._splitby)
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

    def _generate_datasets(self) -> tuple[list[np.ndarray], list[tuple[Any, ...]]]:
        datasets = []
        unique_sl: list[tuple[Any, ...]] = []
        for sl, df in self._source.group_by(self._splitby):
            unique_sl.append(sl)
            datasets.append(df[self._y])
        return datasets, unique_sl

    def _generate_labels(self):
        """Generate the tick positions, labels and the axis label."""
        _agged_by = _shared.unique_tuple(self._color_by.by, self._hatch_by.by)
        _nagged = 0
        for each in reversed(self._offset_by.by):
            if each in _agged_by:
                _nagged += 1
            else:
                break

        # If all the offset columns are redundantly categorized by color or hatch,
        # then all the labels should be shown.
        if _nagged == len(self._offset_by.by):
            _nagged = 0

        # group positions by aggregated labels
        label_to_pos: dict[str, list[float]] = {}
        for p, lbl in self._offset_by.iter_ticks(self._labels, self._splitby):
            label_agged = "\n".join(lbl[: len(lbl) - _nagged])
            if label_agged in label_to_pos:
                label_to_pos[label_agged].append(p)
            else:
                label_to_pos[label_agged] = [p]
        # compute the mean position for each aggregated label
        pos: list[float] = []
        labels: list[str] = []
        for label, pos_list in label_to_pos.items():
            pos.append(np.mean(pos_list))
            labels.append(label)

        offset_labels = self._offset_by.by[: len(self._offset_by.by) - _nagged]
        return pos, labels, offset_labels


class DFViolinPlot(
    _shared.DataFrameLayerWrapper[_lg.ViolinPlot, _DF],
    _BoxLikeMixin,
    Generic[_DF],
):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        offset: str | tuple[str, ...],
        value: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        shape: str = "both",
        backend: str | Backend | None = None,
    ):
        if isinstance(offset, str):
            offset = (offset,)
        _BoxLikeMixin.__init__(self, source, offset, value, color, hatch)
        arrays, self._labels = self._generate_datasets()
        x = self._offset_by.generate(self._labels, self._splitby)
        base = _lg.ViolinPlot.from_arrays(
            x, arrays, name=name, orient=orient, shape=shape, extent=extent,
            backend=backend,
        )  # fmt: skip
        super().__init__(base, source)
        if color is not None:
            self.with_color(color)
        if hatch is not None:
            self.with_hatch(hatch)

    @classmethod
    def from_table(
        cls,
        df: _DF,
        offset: tuple[str, ...],
        value: str,
        color: str | None = None,
        hatch: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        shape: str = "both",
        backend: str | Backend | None = None,
    ) -> DFViolinPlot[_DF]:
        src = parse(df)
        self = DFViolinPlot(
            src, offset, value, orient=orient, name=name, extent=extent,
            color=color, hatch=hatch, shape=shape, backend=backend
        )  # fmt: skip
        return self

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
        source: DataFrameWrapper[_DF],
        offset: str | tuple[str, ...],
        value: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ):
        _BoxLikeMixin.__init__(self, source, offset, value, color, hatch)
        arrays, self._labels = self._generate_datasets()
        x = self._offset_by.generate(self._labels, self._splitby)
        base = _lg.BoxPlot.from_arrays(
            x,
            arrays,
            name=name,
            orient=orient,
            capsize=capsize,
            extent=extent,
            backend=backend,
        )
        super().__init__(base, source)
        base.with_edge(color=theme.get_theme().foreground_color)
        if color is not None:
            self.with_color(color)
        if hatch is not None:
            self.with_hatch(hatch)

    @classmethod
    def from_table(
        cls,
        df: _DF,
        offset: tuple[str, ...],
        value: str,
        color: str | None = None,
        hatch: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ) -> DFBoxPlot[_DF]:
        src = parse(df)
        self = DFBoxPlot(
            src, offset, value, orient=orient, name=name, color=color, hatch=hatch,
            capsize=capsize, extent=extent, backend=backend
        )  # fmt: skip
        return self

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
        arrays, _ = self._generate_datasets()
        est = [est_func(arr) for arr in arrays]
        self._set_estimation_values(est)
        return self

    def _update_error(
        self,
        err_func: Callable[[np.ndarray], tuple[float, float]],
    ) -> Self:
        arrays, _ = self._generate_datasets()
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
        source: DataFrameWrapper[_DF],
        offset: str | tuple[str, ...],
        value: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        backend: str | Backend | None = None,
    ):
        _BoxLikeMixin.__init__(self, source, offset, value, color, hatch)
        arrays, self._labels = self._generate_datasets()
        x = self._offset_by.generate(self._labels, self._splitby)
        base = _lg.LabeledPlot.from_arrays(
            x, arrays, name=name, orient=orient, capsize=capsize, backend=backend,
        )  # fmt: skip
        super().__init__(base, source)
        base.with_edge(color=theme.get_theme().foreground_color)
        self._orient = orient
        if color is not None:
            self.with_color(color)
        if hatch is not None:
            self.with_hatch(hatch)

    @classmethod
    def from_table(
        cls,
        df: _DF,
        offset: tuple[str, ...],
        value: str,
        color: str | None = None,
        hatch: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        backend: str | Backend | None = None,
    ) -> DFPointPlot[_DF]:
        src = parse(df)
        self = DFPointPlot(
            src, offset, value, orient=orient, name=name, color=color, hatch=hatch,
            capsize=capsize, backend=backend
        )  # fmt: skip
        return self

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
        source: DataFrameWrapper[_DF],
        offset: str | tuple[str, ...],
        value: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ):
        _BoxLikeMixin.__init__(self, source, offset, value, color, hatch)
        arrays, self._labels = self._generate_datasets()
        x = self._offset_by.generate(self._labels, self._splitby)
        base = _lg.LabeledBars.from_arrays(
            x, arrays, name=name, orient=orient, capsize=capsize, extent=extent,
            backend=backend,
        )  # fmt: skip
        super().__init__(base, source)
        base.with_edge(color=theme.get_theme().foreground_color)
        self._orient = orient
        if color is not None:
            self.with_color(color)
        if hatch is not None:
            self.with_hatch(hatch)

    @classmethod
    def from_table(
        cls,
        df: _DF,
        offset: tuple[str, ...],
        value: str,
        color: str | None = None,
        hatch: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ) -> DFBarPlot[_DF]:
        src = parse(df)
        self = DFBarPlot(
            src, offset, value, orient=orient, name=name, color=color, hatch=hatch,
            capsize=capsize, extent=extent, backend=backend,
        )  # fmt: skip
        return self

    @property
    def orient(self) -> Orientation:
        return self._base_layer.bars.orient

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
