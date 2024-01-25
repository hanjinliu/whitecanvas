from __future__ import annotations

from whitecanvas.theme._dataclasses import Theme

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


def default(attr: str, value):
    if value is not None:
        return value
    out = _DEFAULT_THEME
    for a in attr.split("."):
        out = getattr(out, a)
    return out
