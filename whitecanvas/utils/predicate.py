from __future__ import annotations


def not_starts_with_underscore(name: str) -> bool:
    return not name.startswith("_")


not_starts_with_underscore.__repr__ = lambda: "<not starts with `_`>"
