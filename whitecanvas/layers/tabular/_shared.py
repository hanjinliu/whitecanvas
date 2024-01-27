from __future__ import annotations

from typing import Any, Generic, Iterable, TypeVar

from whitecanvas.layers._base import Layer, LayerWrapper
from whitecanvas.layers.tabular._df_compat import DataFrameWrapper

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
            out.extend(cv.columns)
    return tuple(out)


def unique_tuple(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    b_filt = tuple(x for x in b if x not in a)
    return a + b_filt
