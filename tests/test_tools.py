from whitecanvas import new_canvas
from whitecanvas import tools
from unittest.mock import MagicMock

def test_line_selector(backend: str):
    canvas = new_canvas(backend)
    mock = MagicMock()
    selector = tools.line_selector(canvas)
    selector.changed.connect(mock)
    canvas.mouse.moved.emulate_drag([[0, 0], [0, 1], [1, 1]])
    assert mock.call_count == 1
    mock.assert_called_with(((0, 0), (1, 1)))

    canvas = new_canvas(backend)
    mock = MagicMock()
    selector = tools.line_selector(canvas, tracking=True)
    selector.changed.connect(mock)
    canvas.mouse.moved.emulate_drag([[0, 0], [0, 1], [1, 1]])
    assert mock.call_count == 2

def test_other_selectors(backend: str):
    canvas = new_canvas(backend)
    mock = MagicMock()
    selector = tools.rect_selector(canvas)
    selector.changed.connect(mock)
    canvas.mouse.moved.emulate_drag([[0, 0], [0, 1], [1, 1]])
    assert mock.call_count == 1
