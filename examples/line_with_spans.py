import numpy as np

from whitecanvas import new_canvas


def main():
    canvas = new_canvas(backend="matplotlib:qt")
    rng = np.random.default_rng(12345)
    x = np.arange(200)
    y = np.concatenate(
        [
            rng.normal(size=50),
            rng.normal(size=30, loc=2.3),
            rng.normal(size=50),
            rng.normal(size=40, loc=2.3),
            rng.normal(size=30),
        ],
    )
    canvas.add_line(x, y, width=1.5, color="black")
    canvas.add_spans([[50, 80], [130, 170]], color="red", alpha=0.3)
    canvas.show(block=True)

if __name__ == "__main__":
    main()
