# This example shows how to efficiently use scipy.optimize.curve_fit

from whitecanvas import new_canvas
from scipy.optimize import curve_fit
import numpy as np

def sample_data():
    x = np.arange(20)
    y = 3.7 * np.exp(-x * 0.27) + np.random.normal(size=20, scale=0.4) + 0.3
    return x, y

def main():
    canvas = new_canvas(backend="matplotlib:qt")

    # add raw data
    x, y = sample_data()
    canvas.add_markers(x, y, color="gray", name="raw data")

    # fitting
    def model(x, a, tau, b):
        return a * np.exp(-x / tau) + b

    params, _ = curve_fit(model, x, y, p0=[2, 1, 0])

    # add the fitting curve
    canvas.add_infcurve(model, color="red", name="fit", width=2).with_params(*params)

    canvas.show(block=True)

if __name__ == "__main__":
    main()
