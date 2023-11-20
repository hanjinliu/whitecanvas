from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Theme:
    fontfamily: str = "sans-serif"
    fontsize: int = 11
    foreground_color: str = "#000000"
    background_color: str = "#FFFFFF"
    palette = "tab10"


_DEFAULT_THEME = Theme()


def get_theme(name: str | None = None) -> Theme:
    # TODO: customizable themes
    if name is None:
        return _DEFAULT_THEME
    return _DEFAULT_THEME
