from __future__ import annotations

from typing import Callable, Sequence, TypeVar

_T = TypeVar("_T")


def make_sorter_manual(order: Sequence[_T]) -> Callable[[Sequence[_T]], Sequence[_T]]:
    if len(order) != len(set(order)):
        raise ValueError(f"Order contains duplicate values: {order!r}")

    def _sort_manual(x: Sequence[_T]) -> Sequence[_T]:
        out: list[_T] = []
        input_categories = set(x)
        for each in order:
            if each in input_categories:
                out.append(each)
                input_categories.discard(each)
        if input_categories:
            raise ValueError(
                f"Categories {input_categories!r} are not in the given order "
                f"{order!r}"
            )
        return order

    return _sort_manual


def sorter_ascending(x: Sequence[_T]) -> Sequence[_T]:
    return sorted(x)


def sorter_descending(x: Sequence[_T]) -> Sequence[_T]:
    return sorted(x, reverse=True)
