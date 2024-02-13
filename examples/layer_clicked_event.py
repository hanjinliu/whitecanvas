from whitecanvas import new_canvas

def main():
    canvas = new_canvas("plotly")
    line0 = canvas.add_line([0, 1, 0, 1], color="blue", width=3)
    line1 = canvas.add_line([1, 0, 1, 0], color="blue", width=3)
    markers = (
        canvas.add_markers([2, 2, 2, 2], color="blue", size=20)
        .with_face_multi()
    )
    @line0.events.clicked.connect
    def on_line_clicked(i: int):
        line0.color = "red"

    @line1.events.clicked.connect
    def on_line_clicked(i: int):
        line1.color = "red"

    @markers.events.clicked.connect
    def on_markers_clicked(i: int):
        colors = markers.face.color
        colors[i] = [1, 0, 0, 1]
        markers.face.color = colors
    canvas.show(block=True)

if __name__ == "__main__":
    main()
