from __future__ import annotations


def not_starts_with_underscore(name: str) -> bool:
    return not name.startswith("_")
