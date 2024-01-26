from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from whitecanvas.theme._dataclasses import Theme

_EXISTING_THEMES = {
    "light": Theme(),
    "dark": Theme(
        foreground_color="#FFFFFF", background_color="#000000", palette="tab10_light"
    ),
}
_DEFAULT_THEME = _EXISTING_THEMES["light"]


def get_theme(name: str | Theme | None = None) -> Theme:
    """
    Get a theme by name.

    >>> import whitecanvas as wc
    >>> theme = wc.theme.get_theme()  # get the default theme
    >>> theme = wc.theme.get_theme("dark")  # get the dark theme
    """
    if isinstance(name, Theme):
        return name
    if name is None:
        return _DEFAULT_THEME
    return _EXISTING_THEMES[name]


def _default(attr: str, value):
    if value is not None:
        return value
    out = _DEFAULT_THEME
    for a in attr.split("."):
        out = getattr(out, a)
    return out


@contextmanager
def context(name: str | Theme | None = None) -> Generator[Theme, None, None]:
    """
    Temporarily change the default theme in this context.

    >>> import whitecanvas as wc
    >>> with wc.theme.context("dark") as theme:
    ...     theme.font.color = "yellow"  # change the theme parameters
    ...     # do something with the dark theme
    >>> # back to the default theme
    """
    global _DEFAULT_THEME
    theme = get_theme(name).copy()
    _default_theme = _DEFAULT_THEME
    theme._set_mutable(True)
    _DEFAULT_THEME = theme
    try:
        yield theme
    finally:
        _DEFAULT_THEME = _default_theme


def update_default(name: str | Theme | None = None) -> _UpdateDefault:
    """
    Update the default theme.

    This function can be directly called to update the default theme by name.

    >>> import whitecanvas as wc
    >>> wc.theme.update_default("dark")  # change the default theme

    You can also use it as a context manager to further change the theme parameters.
    The updated parameters persist after the context exits.

    >>> with wc.theme.update_default("dark") as theme:
    ...     theme.font.color = "yellow"  # change the theme parameters
    >>> # the default theme is now the dark theme with yellow font
    """
    return _UpdateDefault(name)


class _UpdateDefault:
    def __init__(self, name: str | Theme | None = None):
        global _DEFAULT_THEME
        self._theme = get_theme(name)
        self._default_theme = _DEFAULT_THEME.copy()
        _DEFAULT_THEME = self._theme

    def __enter__(self):
        self._theme._set_mutable(True)
        return self._theme

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _DEFAULT_THEME
        if exc_type is not None:
            _DEFAULT_THEME = self._default_theme
        self._theme._set_mutable(False)
