# This example shows how to define custom hover texts for markers.

from whitecanvas import new_canvas


def main():
    canvas = new_canvas(backend="matplotlib:qt")

    canvas.add_markers(
        [0, 1, 2], [0, 1, 2], color="red"
    ).with_hover_text(["first", "second", "third"])

    canvas.add_markers(
        [0, 1, 2], [1, 2, 3], color="blue"
    ).with_hover_template(
        "x: {x:.2f}, y: {y:.2f}, index: {i},\ncustom_data: {c}",
        extra=dict(c=[123, 234, 345]),
    )

    canvas.show(block=True)

if __name__ == "__main__":
    main()
