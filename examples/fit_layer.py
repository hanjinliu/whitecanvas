# Use the `fit` function to fit a model to data

import numpy as np
from whitecanvas import new_canvas

def main():
    canvas = new_canvas(backend="vispy:qt")
    x = np.linspace(-1, 1, 100)
    y = x ** 2 / 2 + 1 + np.random.normal(0, 0.1, 100)
    markers = canvas.add_markers(x, y, color="lightgray")
    fit_1 = canvas.fit(markers).linear(color="blue", width=2.5)
    fit_2 = canvas.fit(markers).polynomial(2, color="red", width=2.5)

    print(fit_1, "is linear regression.")
    print(fit_2, "is polynomial regression of degree=2.")
    canvas.show(block=True)

if __name__ == "__main__":
    main()
