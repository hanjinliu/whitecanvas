# In this example, two mouse events are connected to the canvas.
# 1. Dragging the left mouse button without any keyboard modifiers draws a line.
# 2. Dragging the left mouse button with the shift key held down draws a span.

import numpy as np

from whitecanvas import new_canvas
from whitecanvas.types import MouseEvent


def main():
    canvas = new_canvas("matplotlib:qt")
    x, y = np.indices((10, 10)).reshape(2, -1)
    canvas.add_markers(x, y, color="gray")

    @canvas.mouse.moved.connect
    def _add_line(e: MouseEvent):
        if e.button != "left" or e.modifiers != ():
            return
        pos0 = e.pos
        line = canvas.add_line(np.array([pos0, pos0]), color="blue")
        dragged = False
        while e.type != "release":
            data = np.array([pos0, e.pos])
            line.data = data
            yield
            dragged = True
        if not dragged:
            canvas.layers.remove(line)

    @canvas.mouse.moved.connect
    def _add_spans(e: MouseEvent):
        if e.button != "left" or e.modifiers != ("shift",):
            return
        x0 = e.pos[0]
        span = canvas.add_spans([[x0, x0]], color="pink")
        dragged = False
        while e.type != "release":
            data = np.array([[x0, e.pos[0]]])
            span.data = data
            yield
            dragged = True
        if not dragged:
            canvas.layers.remove(span)

    canvas.show(block=True)

if __name__ == "__main__":
    main()
