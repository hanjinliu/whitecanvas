from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar

import numpy as np
from numpy.typing import NDArray

from whitecanvas.backend import Backend
from whitecanvas.layers._base import Layer, LayerWrapper
from whitecanvas.layers._deserialize import construct_layer
from whitecanvas.layers.tabular import _jitter
from whitecanvas.layers.tabular._df_compat import DataFrameWrapper, parse

if TYPE_CHECKING:
    from typing_extensions import Self

_L = TypeVar("_L", bound="Layer")
_DF = TypeVar("_DF")


class DataFrameLayerWrapper(LayerWrapper[_L], Generic[_L, _DF]):
    def __init__(self, base: _L, source: DataFrameWrapper[_DF]):
        super().__init__(base)
        self._source = source

    @property
    def data(self) -> _DF:
        """The internal dataframe."""
        return self._source.get_native()

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        from whitecanvas.layers.tabular._df_compat import from_dict

        base, source = d["base"], d["source"]
        if isinstance(base, dict):
            base = construct_layer(base, backend=backend)
        if isinstance(source, dict):
            source = from_dict(source)
        return cls(base=base, source=source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "base": self._base_layer.to_dict(),
            "source": self._source,
        }


class ColumnOrValue:
    def __init__(self, by, df: DataFrameWrapper[_DF]):
        if isinstance(by, str):
            if by in df.iter_keys():
                self._is_columns = True
                self._value = (by,)
            else:
                self._is_columns = False
                self._value = by
        elif hasattr(by, "__iter__"):
            self._is_columns = all(isinstance(each, str) for each in by)
            if self._is_columns:
                columns = set(df.iter_keys())
                for each in by:
                    if each not in columns:
                        raise ValueError(f"{each!r} is not a column.")
                self._value = tuple(by)
            else:
                self._value = by
        else:
            self._is_columns = False
            self._value = by

    @property
    def is_column(self) -> bool:
        """True if the value is column name(s)."""
        return self._is_columns

    @property
    def value(self) -> Any:
        """Return the value."""
        return self._value

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the column name(s)."""
        if self._is_columns:
            return self._value
        else:
            raise ValueError("The value is not a column name(s).")


def join_columns(
    *objs: Any | Iterable[Any] | None,
    source: DataFrameWrapper[_DF],
) -> tuple[str, ...]:
    """Join objects if it is a column name of the source data frame."""
    out = []
    for obj in objs:
        if obj is None:
            continue
        cv = ColumnOrValue(obj, source)
        if cv.is_column:
            for each in cv.columns:
                if each not in out:
                    out.append(each)
    return tuple(out)


def unique_tuple(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    b_filt = tuple(x for x in b if x not in a)
    return a + b_filt


def norm_dodge(
    source: DataFrameWrapper[_DF],
    offset: str | tuple[str, ...],
    *args: str | tuple[str, ...] | None,
    dodge: str | tuple[str, ...] | bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(offset, str):
        offset = (offset,)
    if isinstance(dodge, bool):
        if dodge:
            _all = join_columns(*args, source=source)
            dodge = tuple(c for c in _all if c not in offset)
        else:
            dodge = ()
    elif isinstance(dodge, str):
        dodge = (dodge,)
    else:
        dodge = tuple(dodge)
    splitby = join_columns(offset, *args, dodge, source=source)
    return splitby, dodge


def norm_dodge_markers(
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
            _all = join_columns(color, hatch, source=source)
            dodge = tuple(c for c in _all if c not in offset)
        else:
            dodge = ()
    elif isinstance(dodge, str):
        dodge = (dodge,)
    else:
        dodge = tuple(dodge)
    splitby = join_columns(offset, dodge, source=source)
    return splitby, dodge


def list_to_df(arr: list[tuple], columns: list[str]) -> DataFrameWrapper[dict]:
    """
    Convert a list of tuples to a DataFrame.

    >>> list_to_df([(1, 2), (3, 4)], ["a", "b"])
    # Out: DataFrameWrapper({"a": [1, 3], "b": [2, 4]})
    """
    out = {c: [] for c in columns}
    for row in arr:
        for c, v in zip(columns, row):
            out[c].append(v)
    return parse(out)


def resolve_stacking(
    df: DataFrameWrapper[_DF],
    x: str | _jitter.JitterBase,
    y: str | _jitter.JitterBase,
    stackby: tuple[str, ...],
) -> tuple[NDArray[np.number], NDArray[np.number], NDArray[np.number]]:
    if isinstance(x, _jitter.JitterBase):
        xj = x
    else:
        xj = _jitter.IdentityJitter(x)
    if isinstance(y, _jitter.JitterBase):
        yj = y
    else:
        yj = _jitter.IdentityJitter(y)
    # pre-calculate all the possible xs
    all_x = xj.map(df)

    def _hash_rule(x: float) -> int:
        return int(round(x * 1000))

    ycumsum = {_hash_rule(_x): 0.0 for _x in all_x}
    x0 = list[NDArray[np.number]]()
    y0 = list[NDArray[np.number]]()
    b0 = list[NDArray[np.number]]()
    for _, sub in df.group_by(stackby):
        this_x = xj.map(sub)
        this_h = yj.map(sub)
        bottom = []
        for _x, _h in zip(this_x, this_h):
            _x_hash = _hash_rule(_x)
            dy = ycumsum[_x_hash]
            bottom.append(dy)
            ycumsum[_x_hash] += _h
        x0.append(this_x)
        y0.append(this_h)
        b0.append(bottom)
    x0 = np.concatenate(x0)
    y0 = np.concatenate(y0)
    b0 = np.concatenate(b0)
    return x0, y0, b0
