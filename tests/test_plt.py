import numpy as np

import whitecanvas.plot as plt


def test_functions():
    arr = np.arange(10)
    plt.figure(size=(400, 300))
    plt.line(arr)
    plt.markers(arr)
    plt.bars(arr, arr / 2)
    plt.band(arr, arr / 2, arr * 2)
    plt.errorbars(arr, arr / 2, arr * 2)
    plt.hist(arr)
    plt.spans([[4, 10], [5, 14]])
    plt.infcurve(lambda x: x ** 2)
    plt.infline((0, 0), 40)
    plt.legend()
    plt.xlim()
    plt.xlim(0, 10)
    plt.ylim()
    plt.ylim(0, 10)
    plt.title("Title")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks()
    plt.xticks([0, 3, 6, 9])
    plt.xticks([0, 3, 6, 9], ["a", "b", "c", "d"])
    plt.yticks()
    plt.yticks([0, 3, 6, 9])
    plt.yticks([0, 3, 6, 9], ["a", "b", "c", "d"])
    plt.show()
