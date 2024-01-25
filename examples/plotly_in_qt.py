import numpy as np

from whitecanvas import plot as plt


def main():
    canvas = plt.figure("plotly:qt")
    x = np.linspace(0, 4, 200)
    times = np.linspace(0, 4, 10)
    multidim = canvas.dims.in_axes("time")
    multidim.add_line(x, [np.sin(x*3 - x0) for x0 in times])
    multidim.add_text([0], [0.8], [[f"t = {t:.2f}"] for t in times], size=28)
    canvas.show(block=True)

if __name__ == "__main__":
    main()
