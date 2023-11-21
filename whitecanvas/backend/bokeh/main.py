from whitecanvas import new_canvas
import numpy as np


def main():
    canvas = new_canvas("bokeh")
    canvas.add_line(np.random.random(100), color="red", width=2)

    @canvas.x.lim_changed.connect
    def _changed(lim):
        print(lim)

    canvas.show()


main()
