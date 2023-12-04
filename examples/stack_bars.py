from whitecanvas import new_canvas

def main():
    canvas = new_canvas(backend="matplotlib:qt")
    x = [1, 2, 3, 4, 5]
    y0 = [2, 3, 2, 1, 2]
    y1 = [3, 2, 1, 2, 3]
    y2 = [0, 1, 2, 3, 4]
    b0 = canvas.add_bars(x, y0)
    b1 = canvas.stack_over(b0).add(y1)
    b2 = canvas.stack_over(b1).add(y2)
    canvas.show(block=True)

if __name__ == "__main__":
    main()
