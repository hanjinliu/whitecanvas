import numpy as np
from whitecanvas import new_canvas
from whitecanvas import tools
from unittest.mock import MagicMock

def test_line_selector(backend: str):
    canvas = new_canvas(backend)
    mock = MagicMock()
    selector = tools.line_selector(canvas)
    selector.changed.connect(mock)
    canvas.mouse.emulate_drag([[0, 0], [0, 1], [1, 1]])
    assert mock.call_count == 1
    mock.assert_called_with(((0, 0), (1, 1)))

    canvas = new_canvas(backend)
    mock = MagicMock()
    selector = tools.line_selector(canvas, tracking=True)
    selector.changed.connect(mock)
    canvas.mouse.emulate_drag([[0, 0], [0, 1], [1, 1]])
    assert mock.call_count == 2

def test_rect_selector(backend: str):
    canvas = new_canvas(backend)
    mock = MagicMock()
    selector = tools.rect_selector(canvas)
    selector.changed.connect(mock)
    canvas.mouse.emulate_drag([[0, 0], [0, 1], [1, 1]])
    assert mock.call_count == 1

def test_xspan_selector(backend: str):
    canvas = new_canvas(backend)
    mock = MagicMock()
    selector = tools.xspan_selector(canvas)
    selector.changed.connect(mock)
    canvas.mouse.emulate_drag([[0, 0], [1, 0]])
    assert mock.call_count == 1
    mock.assert_called_with((0, 1))

def test_yspan_selector(backend: str):
    canvas = new_canvas(backend)
    mock = MagicMock()
    selector = tools.yspan_selector(canvas)
    selector.changed.connect(mock)
    canvas.mouse.emulate_drag([[0, 0], [0, 1]])
    assert mock.call_count == 1
    mock.assert_called_with((0, 1))

def test_lasso_selector(backend: str):
    canvas = new_canvas(backend)
    mock = MagicMock()
    selector = tools.lasso_selector(canvas)
    selector.changed.connect(mock)
    canvas.mouse.emulate_drag([[0, 0], [0, 1], [1, 1], [1, 0]])
    assert mock.call_count == 1
    xy = np.array([(0, 0), (0, 1), (1, 1), (1, 0)])
    arg = mock.call_args[0][0]
    assert np.allclose(arg[0], xy[:, 0])
    assert np.allclose(arg[1], xy[:, 1])
