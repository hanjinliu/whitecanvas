from whitecanvas import new_canvas


def main():
    canvas = new_canvas(backend="matplotlib")
    x = [1, 2, 3, 4, 5]
    y0 = [2, 3, 2, 1, 2]
    y1 = [3, 2, 1, 2, 3]
    canvas.add_line(x, y0, width=3, color="red")
    canvas.add_line(x, y1, width=3, color="blue", style="--")
    canvas.show()

if __name__ == "__main__":
    main()
