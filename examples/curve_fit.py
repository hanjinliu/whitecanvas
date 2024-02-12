# This example shows how to efficiently use scipy.optimize.curve_fit

import numpy as np
from scipy.optimize import curve_fit

from whitecanvas import new_canvas


def sample_data(tau: float, a: float, b: float, size: int = 40):
    x = np.arange(size)
    y = a * np.exp(-x / tau) + np.random.normal(size=size, scale=a*0.1) + b
    return x, y

def main():
    np.random.seed(1462)
    canvas = new_canvas(backend="matplotlib:qt")

    # tau, a, b, size
    params_true = [
        (9.1, 3.6, 0.46, 40),
        (6.8, 3.0, 0.21, 48),
        (7.6, 4.0, 0.58, 32)
    ]

    # fitting model
    def model(x, a, tau, b):
        return a * np.exp(-x / tau) + b

    for p in params_true:
        # add raw data
        x, y = sample_data(*p)
        line = canvas.add_line(x, y, alpha=0.25)

        # add the fitting curve
        params, _ = curve_fit(model, x, y, p0=[2, 1, 0])
        (
            canvas
            .add_infcurve(model, color=line.color, alpha=1)
            .update_params(*params)
            .with_hover_text("a={:.3g}, tau={:.3g}, b={:.3g}".format(*params))
        )

    canvas.update_labels(
        x="time [sec]", y="intensity [a.u.]", title="Fitting results"
    ).show(block=True)

if __name__ == "__main__":
    main()
