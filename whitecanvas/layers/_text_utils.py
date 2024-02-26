from __future__ import annotations

from whitecanvas.types import XYData


def norm_label_text(strings: str | list[str], data: XYData) -> list[str]:
    if not isinstance(strings, str):
        strings = list(strings)
        if len(strings) != data.x.size:
            raise ValueError(
                f"Length of strings ({len(strings)}) does not match data size "
                f"({data.x.size})"
            )
        return strings
    if "{x" in strings or "{y" in strings or "{i" in strings:
        try:
            strings.format(x=0.0, y=0.0, i=0)  # dry run
        except Exception:
            strings = [strings] * data.x.size
        else:
            strings = [
                strings.format(x=x, y=y, i=i) for i, (x, y) in enumerate(zip(*data))
            ]
    else:
        strings = [strings] * data.x.size
    return strings
