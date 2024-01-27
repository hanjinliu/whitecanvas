import whitecanvas as wc
from ._utils import assert_color_equal
import pytest


def test_context():
    with wc.theme.context("dark") as theme:
        theme.foreground_color = "yellow"
        assert_color_equal(theme.foreground_color, "yellow")
        canvas = wc.new_canvas("matplotlib:qt")
        assert_color_equal(canvas.x.color, "yellow")
    canvas = wc.new_canvas("matplotlib:qt")
    assert_color_equal(canvas.x.ticks.color, "black")

def test_validator():
    theme = wc.theme.get_theme()

    with pytest.raises(TypeError):
        theme.font.color = 1
    with pytest.raises(TypeError):
        theme.canvas_size = (-1, 2)
    with pytest.raises(Exception):
        theme.line.style = "not-a-style"

def test_update():
    wc.theme.update_default("dark")
    assert_color_equal(wc.theme.get_theme().background_color, "black")

    with wc.theme.update_default("light") as theme:
        theme.background_color = "#EEEEEE"
    assert_color_equal(wc.theme.get_theme().background_color, "#EEEEEE")
