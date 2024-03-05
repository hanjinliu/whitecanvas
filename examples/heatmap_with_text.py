import numpy as np
from whitecanvas import new_canvas

def main():
    canvas = new_canvas("pyqtgraph:qt")
    rng = np.random.default_rng(14872)
    data = rng.poisson(4, (8, 10))
    heatmap = canvas.add_heatmap(data, cmap="inferno").with_text()
    canvas.show(block=True)

if __name__ == "__main__":
    main()
