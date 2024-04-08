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

def test_enabled():
    canvas = new_canvas("mock")
    mock = MagicMock()
    selector = tools.line_selector(canvas)
    selector.changed.connect(mock)
    assert selector.enabled
    mock.assert_not_called()
    canvas.mouse.emulate_drag([[0, 0], [0, 1], [1, 1]])
    mock.assert_called_once()
    mock.reset_mock()
    selector.enabled = False
    assert not selector.enabled
    canvas.mouse.emulate_drag([[0, 0], [0, 1], [1, 1]])
    mock.assert_not_called()

def test_tracking():
    canvas = new_canvas("mock")
    mock = MagicMock()
    selector = tools.line_selector(canvas, tracking=True)
    selector.changed.connect(mock)
    mock.assert_not_called()
    canvas.mouse.emulate_drag([[0, 0], [0, 1], [1, 1]])
    assert mock.call_count == 2

def test_point_contained():
    canvas = new_canvas("mock")
    with tools.rect_selector(canvas) as sel:
        canvas.mouse.emulate_drag([[0, 0], [1, 1]])
        assert sel.contains_point(0.5, 0.5)
        assert not sel.contains_point(1.5, 1.5)
        assert np.all(sel.contains_points([(0.2, 0.8), (1.3, 0.8)]) == [True, False])
    with tools.xspan_selector(canvas) as sel:
        canvas.mouse.emulate_drag([[0, 0], [1, 0]])
        assert sel.contains_point(0.5, 0)
        assert not sel.contains_point(1.5, 0)
        assert np.all(sel.contains_points([(0.2, 100), (1.3, -100)]) == [True, False])
    with tools.yspan_selector(canvas) as sel:
        canvas.mouse.emulate_drag([[0, 0], [0, 1]])
        assert sel.contains_point(0, 0.5)
        assert not sel.contains_point(0, 1.5)
        assert np.all(sel.contains_points([(100, 0.2), (-100, 1.3)]) == [True, False])
    with tools.lasso_selector(canvas) as sel:
        canvas.mouse.emulate_drag([[0, 0], [0, 1], [2, 1], [2, 2], [1, 2], [1, 0]])
        assert sel.contains_point([0.5, 0.5])
        assert not sel.contains_point(1.5, 0.5)
        assert np.all(sel.contains_points([(1.5, 1.4), (0.2, 1.8)]) == [True, False])
