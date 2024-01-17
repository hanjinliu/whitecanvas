# This example shows how to use the `with_yfill` method to fill the area between
# a line and the x-axis.

import numpy as np

from whitecanvas import new_canvas


def main():
    canvas = new_canvas("matplotlib:qt")
    x = np.linspace(0, 1, 200)
    y = 1 / (1 + x**2)
    canvas.add_line(x, y, color="black", width=2)
    canvas.add_line(x[50:120], y[50:120], color="red", width=4).with_yfill(alpha=0.3)

    canvas.show(block=True)

if __name__ == "__main__":
    main()
