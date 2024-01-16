import numpy as np

import whitecanvas.plot as plt


def test_functions():
    arr = np.arange(10)
    plt.figure("matplotlib", size=(400, 300))
    plt.line(arr)
    plt.markers(arr)
    plt.bars(arr, arr / 2)
    plt.band(arr, arr / 2, arr * 2)
    plt.errorbars(arr, arr / 2, arr * 2)
    plt.hist(arr)
    plt.spans([[4, 10], [5, 14]])
    plt.infcurve(lambda x: x ** 2)
    plt.infline((0, 0), 40)
    plt.show()
