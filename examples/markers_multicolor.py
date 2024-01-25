import numpy as np

from whitecanvas import new_canvas


def main():
    canvas = new_canvas("matplotlib:qt")
    rng = np.random.default_rng(14632)

    canvas.add_markers(
        rng.random(30) * 10,
        rng.random(30) * 10,
    ).with_face_multi(
        color=rng.random((30, 3)),
    ).with_edge_multi(
        color=rng.random((30, 3)),
        width=rng.random(30) * 2 + 1.0,
    ).with_size_multi(
        rng.random(30) * 30 + 5,
    )

    canvas.show(block=True)

if __name__ == "__main__":
    main()
