from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, TypeVar

import numpy as np
from cmap import Color, Colormap
from numpy.typing import NDArray

from whitecanvas.canvas._palette import ColorPalette
from whitecanvas.types import Hatch, LineStyle, Symbol
from whitecanvas.utils.collections import OrderedSet
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.layers.tabular._df_compat import DataFrameWrapper

_V = TypeVar("_V", bound=Any)
_DF = TypeVar("_DF")


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

    def is_not_const(self) -> bool:
        """Return True if the plan is not a constant plan."""
        return not self.is_const()

    def get_const_value(self) -> _V:
        """Return the constant value if the plan is a constant plan."""
        if self.is_const():
            return self.values[0]
        raise ValueError("The plan is not a constant plan.")

    def generate(
        self,
        labels: list[tuple[Any, ...]],
        by_all: tuple[str, ...],
    ) -> list[_V]:
        # labels = [("a",), ("b",)] ... the all set of unique labels
        # by_all = ("column-0", "column-1")
        indices = [by_all.index(b) for b in self.by]
        size = len(self.values)
        out_lookup: dict[tuple[Any, ...], _V] = {}
        i = 0
        for row in labels:
            row_filt = tuple(row[i] for i in indices)
            if row_filt not in out_lookup:
                out_lookup[row_filt] = self.values[i % size]
                i += 1

        ret = [out_lookup[tuple(_r[i] for i in indices)] for _r in labels]
        return ret

    def create_key_values(
        self,
        values: DataFrameWrapper[_DF],  # the data frame
    ) -> list[tuple[tuple, _V]]:
        """Map dataframe to values of the same size."""
        if self._by:
            series = [values[k] for k in self._by]
        else:
            # constant, no key filter
            return [((), self.values[0])]
        out = []
        i = 0
        for row in OrderedSet(zip(*series)):
            val = self.values[i % len(self.values)]
            out.append((row, val))
            i += 1
        return out

    def map(
        self,
        values: DataFrameWrapper[_DF],  # the data frame
    ) -> Sequence[_V]:
        """Map dataframe to values of the same size."""
        if self._by:
            series = [values[k] for k in self._by]
        else:
            # constant, no key filter
            return [self.values[0]] * len(values)
        out = np.empty(series[0].shape, dtype=object)
        i = 0
        for row in OrderedSet(zip(*series)):
            sl: NDArray[np.bool_] = np.all(
                np.column_stack([a == b for a, b in zip(series, row)]), axis=1
            )
            ntrue = sl.sum()
            val = self.values[i % len(self.values)]
            out[sl] = np.full(ntrue, val, dtype=object)
            i += 1
        return out

    def to_entries(self, df: DataFrameWrapper[_DF]) -> list[tuple[str, _V]]:
        """Prepare legend item entries."""
        if len(df) == 0:
            return []
        values = self.map(df)
        if self.by:
            entries = [
                (", ".join(str(n) for n in key), value)
                for key, value in self.create_key_values(df)
            ]
        else:
            entries = [("", values[0])]
        return entries

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(tuple(data["by"]), [cls._norm_value(v) for v in data["values"]])

    def to_dict(self) -> dict[str, Any]:
        return {
            "by": self.by,
            "values": self.values,
        }


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

    def is_not_const(self) -> bool:
        """Return True if the plan is not a constant plan."""
        return not self.is_const()

    def get_const_value(self) -> _V:
        """Return the constant value if the plan is a constant plan."""
        if isinstance(self._mapper, ConstMap):
            return self._mapper._value
        raise ValueError("The plan is not a constant plan.")

    @classmethod
    def default(cls) -> Self:
        return cls((), cls._default_mapper())

    @classmethod
    @abstractmethod
    def _default_mapper(cls) -> Callable[[dict[str, np.ndarray]], Sequence[_V]]:
        """Default mapper for the plan."""

    def map(
        self,
        values: DataFrameWrapper[_DF],  # the data frame
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        mapper = data.get["mapper"]
        if isinstance(mapper, dict):
            if mapper["type"] == "const":
                mapper = ConstMap(mapper["value"])
            elif mapper["type"] == "ranged":
                mapper = RangedMap.from_dict(mapper)
            elif mapper["type"] == "colormap":
                mapper = ColormapMap.from_dict(mapper)
            else:
                raise ValueError(f"Unknown mapper type: {mapper['type']}")
        return cls(data["on"], mapper)

    def to_dict(self) -> dict[str, Any]:
        return {"on": self._on, "mapper": self._mapper}


class ScalarMapPlan(MapPlan[float]):
    @classmethod
    def from_range(
        cls,
        on: str,
        range: tuple[float, float] | None = None,
        domain: tuple[float, float] | None = None,
    ) -> Self:
        """Add a mapper that maps a range to a value."""
        return cls.from_map((on,), RangedMap(on, range, domain))

    def get_ranged_map(self) -> RangedMap | None:
        if isinstance(self._mapper, RangedMap):
            return self._mapper
        return None


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
        values: DataFrameWrapper[_DF],  # the data frame
    ) -> Sequence[_V]:
        if self._by:
            series = [values[k] for k in self._by]
        else:
            # constant, no key filter
            return [self.values[0]] * len(values)
        out = np.empty((series[0].size, 4), dtype=np.float32)
        i = 0
        for row in OrderedSet(zip(*series)):
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
        return cls.from_map((on,), ColormapMap(on, cmap, clim))

    def get_colormap_map(self) -> ColormapMap | None:
        if isinstance(self._mapper, ColormapMap):
            return self._mapper
        return None


class WidthPlan(ScalarMapPlan):
    @classmethod
    def _default_mapper(cls):
        return ConstMap(1.0)


class SizePlan(ScalarMapPlan):
    @classmethod
    def _default_mapper(cls):
        return ConstMap(12.0)


class AlphaPlan(ScalarMapPlan):
    @classmethod
    def _default_mapper(cls):
        return ConstMap(1.0)


class _SerializableMap(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: ...
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


class ConstMap(_SerializableMap):
    def __init__(self, value):
        self._value = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}<{self._value!r}>"

    def __call__(self, values: dict[str, np.ndarray]) -> Sequence[_V]:
        series = next(iter(values.values()), np.zeros(0))
        return [self._value] * series.size

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(data["value"])

    def to_dict(self) -> dict[str, Any]:
        return {"type": "const", "value": self._value}


class RangedMap(_SerializableMap):
    def __init__(self, on, range=None, domain=None):
        _check_min_max(range)
        _check_min_max(domain)
        self._on = on
        self._range = range
        self._domain = domain

    def __call__(self, values: dict[str, np.ndarray]) -> Sequence[_V]:
        arr = values[self._on]
        if self._domain is None:
            valid = np.isfinite(arr)
            amin, amax = arr[valid].min(), arr[valid].max()
            if amin == amax:
                if self._range is not None:
                    w0, w1 = self._range
                    return np.full(arr.shape, (w0 + w1) / 2)
                else:
                    return np.full(arr.shape, amin)
        else:
            amin, amax = self._domain
        _arr = arr.clip(amin, amax)
        _arr[np.isnan(_arr)] = amin
        if self._range is not None:
            w0, w1 = self._range
            return (_arr - amin) / (amax - amin) * (w1 - w0) + w0
        else:
            return _arr

    def create_samples(self, values: dict[str, np.ndarray]) -> list[tuple[_V, _V]]:
        arr = values[self._on]
        if self._domain is None:
            valid = np.isfinite(arr)
            amin, amax = arr[valid].min(), arr[valid].max()
        else:
            amin, amax = self._domain
        if self._range is not None:
            w0, w1 = self._range
        else:
            w0, w1 = amin, amax

        if w0 == w1:
            return [((w0 + w1) / 2, amin)]
        else:
            xs = np.linspace(0, 1, 3)
            return [((x * (w1 - w0) + w0), x * (amax - amin) + amin) for x in xs]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(data["on"], data.get("range"), data.get("domain"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ranged",
            "on": self._on,
            "range": self._range,
            "domain": self._domain,
        }


class ColormapMap:
    def __init__(self, on, cmap: Colormap, clim=None):
        _check_min_max(clim)
        self._on = on
        self._cmap = cmap
        self._clim = clim

    def __call__(self, values: dict[str, np.ndarray]) -> Sequence[_V]:
        arr = values[self._on]
        valid = np.isfinite(arr)
        if self._clim is None:
            amin, amax = arr[valid].min(), arr[valid].max()
        else:
            amin, amax = self._clim
        if amin == amax:
            color = self._cmap(0.5)
            return np.full((arr.size, 4), np.asarray(color))
        _arr = arr.clip(amin, amax)
        _arr[np.isnan(_arr)] = amin
        return self._cmap((_arr - amin) / (amax - amin))

    def create_samples(self, values: dict[str, np.ndarray]) -> list[tuple[Color, _V]]:
        """Sample colors for the legends."""
        arr = values[self._on]
        valid = np.isfinite(arr)
        if self._clim is None:
            amin, amax = arr[valid].min(), arr[valid].max()
        else:
            amin, amax = self._clim
        if amin == amax:
            color = self._cmap(0.5)
            return [(color, amin)]
        if len(self._cmap.color_stops) > 4:
            xs = np.linspace(0, 1, 4)
        else:
            xs = self._cmap.color_stops
        return [(self._cmap(x), amin * (1 - x) + amax * x) for x in xs]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(data["on"], Colormap(data["cmap"]), data.get("clim"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "colormap",
            "on": self._on,
            "cmap": self._cmap,
            "clim": self._clim,
        }


def _check_min_max(val: tuple[float, float] | None = None):
    if val is None:
        return
    mn, mx = val
    if not (is_real_number(mn) and is_real_number(mx)):
        raise TypeError(f"Must be a tuple of two real numbers, got {val!r}.")
    if mn > mx:
        raise ValueError(f"min must be less than or equal to max, got {mn} and {mx}.")
    return
