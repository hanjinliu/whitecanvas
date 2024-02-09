# This example shows how to display an image when the corresponding marker is picked.

from __future__ import annotations

import numpy as np
from whitecanvas import hgrid

def make_images() -> np.ndarray:
    # prepare sample image data
    images = []
    def sig(x, a):
        return np.exp(x/a)/(1 + np.exp(x/a))

    mean_intensities = sig(np.arange(20) - 10, 2)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    weight = np.exp(-(xx**2 + yy**2)*2)
    for i0 in mean_intensities:
        img = weight * i0 + np.random.normal(scale=0.3, size=(30, 30))
        images.append(img * weight)
    return np.stack(images, axis=0)

def main():
    images = make_images()
    means = np.mean(images, axis=(1, 2))  # calculate mean intensity to plot

    g = hgrid(2, backend="matplotlib:qt")

    # markers to be picked
    markers = (
        g.add_canvas(0)
        .update_labels(x="time", y="intensity")
        .add_markers(means, color="black")
        .with_hover_text([f"{m:.3f}" for m in means])
    )

    # image to be displayed
    img_layer = (
        g.add_canvas(1)
        .update_labels(title="image slice")
        .add_image(images[0], cmap="inferno")
    )

    # connect pick event
    @markers.events.picked.connect
    def _on_pick(indices):
        if len(indices) == 1:
            i = indices[0]
            img_layer.data = images[i]
            img_layer.clim = images[i].min(), images[i].max()

    g.show(block=True)

if __name__ == "__main__":
    main()
