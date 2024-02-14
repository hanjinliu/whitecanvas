from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, TypeVar

import numpy as np
from cmap import Color, Colormap
from numpy.typing import NDArray

from whitecanvas.canvas._palette import ColorPalette
from whitecanvas.layers.tabular._utils import unique
from whitecanvas.types import Hatch, LineStyle, Symbol
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.layers.tabular._df_compat import DataFrameWrapper

_V = TypeVar("_V", bound=Any)


class Plan(ABC, Generic[_V]):
    @abstractmethod
    def generate(
        self,
        vals: np.ndarray,
        by_all: tuple[str, ...],
    ) -> list[_V]:
        """
        Generate values for given labels.

        Length of the returned list must be equal to vals.shape[0].
        """


class CategoricalPlan(Plan[_V]):
    def __init__(self, by: tuple[str, ...]):
        self._by = by

    @property
    def by(self) -> tuple[str, ...]:
        return self._by

    @classmethod
    @abstractmethod
    def default(cls, by: Sequence[str]) -> Self:
        """Create a default plan."""


class CyclicPlan(CategoricalPlan[_V]):
    """
    Plan of cyclic values.

    This object is used to generate values for each category.
    """

    def __init__(self, by: tuple[str, ...], values: list[_V]):
        super().__init__(by)
        self._values = values

    def __repr__(self) -> str:
        cname = type(self).__name__
        return f"{cname}(by={self._by!r}, values={self._values})"

    @property
    def values(self) -> list[_V]:
        return self._values

    @classmethod
    @abstractmethod
    def _default_values(cls) -> list[_V]:
        """Default values for the plan."""

    @classmethod
    @abstractmethod
    def _norm_value(cls, v: Any) -> _V:
        """Normalize the value."""

    @classmethod
    def default(cls) -> Self:
        return cls((), cls._default_values())

    @classmethod
    def new(cls, by, values: list[_V] | None = None):
        """Create a new plan."""
        if values is None:
            values = cls._default_values()
        return cls(tuple(by), values)

    @classmethod
    def from_const(cls, value: _V) -> Self:
        """Create a plan that always returns a constant value."""
        return cls((), values=[cls._norm_value(value)])

    def is_const(self) -> bool:
        """Return True if the plan is a constant plan."""
        return len(self.by) == 0

    def generate(
        self,
        labels: list[tuple[Any, ...]],
        by_all: tuple[str, ...],
    ) -> list[_V]:
        # labels = [("a",), ("b",)] ... the all set of unique labels
        # by_all = ("column-0", "column-1")
        indices = [by_all.index(b) for b in self.by]
        size = len(self.values)
        # filt = _filter_unique(labels, indices)
        out_lookup: dict[tuple[Any, ...], _V] = {}
        i = 0
        for row in labels:
            row_filt = tuple(row[i] for i in indices)
            if row_filt not in out_lookup:
                out_lookup[row_filt] = self.values[i % size]
                i += 1

        ret = [out_lookup[tuple(_r[i] for i in indices)] for _r in labels]
        return ret

    def map(
        self,
        values: DataFrameWrapper,  # the data frame
    ) -> Sequence[_V]:
        if self._by:
            series = [values[k] for k in self._by]
        else:
            # constant, no key filter
            return self.values[0]
        uniques = [unique(ar, axis=None) for ar in series]
        out = np.empty(series[0].shape, dtype=object)
        i = 0
        for row in itertools.product(*uniques):
            sl: NDArray[np.bool_] = np.all(
                np.column_stack([a == b for a, b in zip(series, row)]), axis=1
            )
            ntrue = sl.sum()
            if ntrue > 0:
                val = self.values[i % len(self.values)]
                out[sl] = np.full(ntrue, val, dtype=object)
                i += 1
        return out


class MapPlan(ABC, Generic[_V]):
    def __init__(
        self,
        on: tuple[str, ...],
        mapper: Callable[[dict[str, np.ndarray]], Sequence[_V]],
    ):
        self._on = on
        self._mapper = mapper

    def __repr__(self) -> str:
        cname = type(self).__name__
        if self._on:
            return f"{cname}(on={self._on!r}, mapper={self._mapper})"
        else:
            return f"{cname}(mapper={self._mapper!r})"

    @classmethod
    def from_map(
        cls,
        on: Sequence[str],
        mapper: Callable[[dict[str, np.ndarray]], Sequence[_V]],
    ) -> Self:
        if not isinstance(on, (tuple, list)):
            raise TypeError(f"on must be a sequence, not {type(on)}")
        return cls(tuple(on), mapper)

    @classmethod
    def from_const(cls, value: _V) -> Self:
        """Create a map plan that always returns a constant value."""
        return cls.from_map((), ConstMap(value))

    def is_const(self) -> bool:
        """Return True if the plan is a constant plan."""
        return isinstance(self._mapper, ConstMap)

    @classmethod
    def default(cls) -> Self:
        return cls((), cls._default_mapper())

    @classmethod
    @abstractmethod
    def _default_mapper(cls) -> Callable[[dict[str, np.ndarray]], Sequence[_V]]:
        """Default mapper for the plan."""

    def map(
        self,
        values: DataFrameWrapper,  # the data frame
    ) -> Sequence[_V]:
        """Calculate the values for the input data frame."""
        if self._on:
            input_dict = {k: values[k] for k in self._on}
        else:
            # constant or default size, no key filter
            try:
                k, v = next(iter(values.iter_items()))
                input_dict = {k: v}
            except StopIteration:
                input_dict = {}
        return self._mapper(input_dict)


class ScalarMapPlan(MapPlan[float]):
    @classmethod
    def from_range(
        cls,
        on: str,
        range: tuple[float, float] | None = None,
        domain: tuple[float, float] | None = None,
    ) -> Self:
        """Add a mapper that maps a range to a value."""
        _check_min_max(range)
        _check_min_max(domain)

        def mapper(values: dict[str, np.ndarray]) -> Sequence[_V]:
            arr = values[on]
            if domain is None:
                valid = np.isfinite(arr)
                amin, amax = arr[valid].min(), arr[valid].max()
                if amin == amax:
                    if range is not None:
                        w0, w1 = range
                        return np.full(arr.shape, (w0 + w1) / 2)
                    else:
                        return np.full(arr.shape, amin)
            else:
                amin, amax = domain
            _arr = arr.clip(amin, amax)
            _arr[np.isnan(_arr)] = amin
            if range is not None:
                w0, w1 = range
                return (_arr - amin) / (amax - amin) * (w1 - w0) + w0
            else:
                return _arr

        return cls.from_map((on,), mapper)


class ColorPlan(CyclicPlan[Color]):
    @classmethod
    def _default_values(cls) -> list[Color]:
        return [Color("black")]  # NOTE: will be updated by the canvas palette

    @classmethod
    def _norm_value(cls, v: Any) -> Color:
        return Color(v)

    @classmethod
    def from_palette(
        cls,
        by: Sequence[str],
        palette: ColorPalette | None = None,
        maybe_const: bool = False,
    ) -> ColorPlan:
        if palette is None:
            palette = "tab10"
        palette = ColorPalette(palette)
        colors = palette.nextn(palette.ncolors, update=False)
        if len(colors) == 1 and maybe_const:
            return cls.from_const(colors[0])
        return cls(tuple(by), colors)

    # NOTE: Color instance is detected as a sequence of 4 floats
    #       so we need to override the default mapper
    def map(
        self,
        values: dict[str, np.ndarray],  # the data frame
    ) -> Sequence[_V]:
        if self._by:
            series = [values[k] for k in self._by]
        else:
            # constant, no key filter
            return self.values[0]
        uniques = [unique(ar, axis=None) for ar in series]
        out = np.empty((series[0].size, 4), dtype=np.float32)
        i = 0
        for row in itertools.product(*uniques):
            sl: NDArray[np.bool_] = np.all(
                np.column_stack([a == b for a, b in zip(series, row)]), axis=1
            )
            ntrue = sl.sum()
            if ntrue > 0:
                val = self.values[i % len(self.values)]
                out[sl] = [val] * ntrue
                i += 1
        return out


class StylePlan(CyclicPlan[LineStyle]):
    @classmethod
    def _default_values(cls) -> list[LineStyle]:
        return [
            LineStyle.SOLID,
            LineStyle.DASH,
            LineStyle.DOT,
            LineStyle.DASH_DOT,
        ]

    @classmethod
    def _norm_value(cls, v: Any) -> LineStyle:
        return LineStyle(v)

    def with_choices(self, choices: Sequence[LineStyle]) -> ColorPlan:
        choices = [LineStyle(s) for s in choices]
        return ColorPlan(self._by, choices)


class HatchPlan(CyclicPlan[Hatch]):
    @classmethod
    def _default_values(cls) -> list[Hatch]:
        return [
            Hatch.SOLID,
            Hatch.DIAGONAL_BACK,
            Hatch.HORIZONTAL,
            Hatch.CROSS,
            Hatch.VERTICAL,
            Hatch.DIAGONAL_CROSS,
            Hatch.DOTS,
            Hatch.DIAGONAL_FORWARD,
        ]

    @classmethod
    def _norm_value(cls, v: Any) -> Hatch:
        return Hatch(v)


class SymbolPlan(CyclicPlan[Symbol]):
    @classmethod
    def _default_values(cls) -> list[Symbol]:
        return [
            Symbol.CIRCLE,
            Symbol.SQUARE,
            Symbol.TRIANGLE_UP,
            Symbol.DIAMOND,
            Symbol.TRIANGLE_DOWN,
        ]

    @classmethod
    def _norm_value(cls, v: Any) -> Symbol:
        return Symbol(v)


class ColormapPlan(MapPlan[NDArray[np.float32]]):
    @classmethod
    def _default_mapper(cls):
        return lambda x: np.ones((x.size, 4))

    @classmethod
    def from_colormap(
        cls,
        on: str,
        cmap: Colormap,
        *,
        clim: tuple[float, float] | None = None,
    ) -> Self:
        _check_min_max(clim)

        def mapper(values: dict[str, np.ndarray]) -> Sequence[_V]:
            arr = values[on]
            valid = np.isfinite(arr)
            if clim is None:
                amin, amax = arr[valid].min(), arr[valid].max()
            else:
                amin, amax = clim
            if amin == amax:
                color = cmap(0.5)
                return np.full((arr.size, 4), np.asarray(color))
            _arr = arr.clip(amin, amax)
            _arr[np.isnan(_arr)] = amin
            return cmap((_arr - amin) / (amax - amin))

        return cls.from_map((on,), mapper)


class WidthPlan(ScalarMapPlan):
    @classmethod
    def _default_mapper(cls):
        return lambda _: 1.0


class SizePlan(ScalarMapPlan):
    @classmethod
    def _default_mapper(cls):
        return lambda _: 12.0


class ConstMap:
    def __init__(self, value):
        self._value = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}<{self._value!r}>"

    def __call__(self, values: dict[str, np.ndarray]) -> Sequence[_V]:
        series = next(iter(values.values()), np.zeros(0))
        return [self._value] * series.size


def _check_min_max(val: tuple[float, float] | None = None):
    if val is None:
        return
    mn, mx = val
    if not (is_real_number(mn) and is_real_number(mx)):
        raise TypeError(f"Must be a tuple of two real numbers, got {val!r}.")
    if mn > mx:
        raise ValueError(f"min must be less than or equal to max, got {mn} and {mx}.")
    return
