import numpy as np
from scipy import ndimage as ndi
from whitecanvas import new_canvas, vgrid_nonuniform

def main():
    grid = vgrid_nonuniform(heights=[3, 1], backend="pyqtgraph:qt")
    c0 = grid.add_canvas(0)
    c1 = grid.add_canvas(1)
    image = np.random.random((100,100))
    c0.add_image(image)
    p0 = [20, 80]
    p1 = [50, 60]

    def get_profile(p0, p1):
        length = np.sqrt((p0[0] - p1[0])** 2 + (p0[1] - p1[1])** 2)
        xs = np.linspace(p0[1], p1[1], int(length))
        ys = np.linspace(p0[0], p1[0], int(length))
        return ndi.map_coordinates(image, np.stack([ys, xs], axis=0))

    c0.add_line([p0[1], p1[1]], [p0[0], p1[0]], color="yellow", width=3)
    c1.add_line(get_profile(p0, p1), color="black")
    grid.show(block=True)

if __name__ == "__main__":
    main()
