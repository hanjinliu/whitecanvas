from __future__ import annotations

from typing import Any, Callable


def make_sorter(order: tuple[Any, ...]) -> Callable[[tuple], int]:
    def _sort_in_order(val: tuple[Any, ...]) -> int:
        try:
            return order.index(val)
        except ValueError:
            raise ValueError(
                f"Value {val!r} is in the data but not given in the order."
            )

    return _sort_in_order
