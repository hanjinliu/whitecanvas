import numpy as np
from whitecanvas import new_canvas

def main():
    canvas = new_canvas(backend="matplotlib")
    x = np.arange(30)
    y = np.sin(x / 5)
    yerr = np.random.default_rng(12345).normal(size=30, scale=0.2)

    layer = (
        canvas
        .add_line(x, y, width=2, color="black")
        .with_markers(symbol="o", size=12)
        .with_yerr(yerr, capsize=0.5, color="black")
    )

    print(layer)
    canvas.show()

if __name__ == "__main__":
    main()
