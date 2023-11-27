from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Theme:
    fontfamily: str = "Arial"
    fontsize: int = 11
    foreground_color: str = "#000000"
    background_color: str = "#FFFFFF"
    grid_color: str = "#CCCCCC"
    palette: str = "tab10"


_EXISTING_THEMES = {
    "light": Theme(),
    "dark": Theme(
        foreground_color="#FFFFFF", background_color="#000000", palette="tab10_light"
    ),
}
_DEFAULT_THEME = _EXISTING_THEMES["light"]


def get_theme(name: str | None = None) -> Theme:
    # TODO: customizable themes
    if name is None:
        return _DEFAULT_THEME
    return _EXISTING_THEMES[name]
