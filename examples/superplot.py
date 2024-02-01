import numpy as np

from whitecanvas import new_canvas


def rand(mean: float, n: int) -> list[float]:
    """Generate random data."""
    return np.random.normal(loc=mean, scale=mean / 4, size=n).tolist()

def main():
    # generate some random data
    np.random.seed(174623)
    data = {
        "label": ["Control"] * 50 + ["Treatment"] * 50,
        "value": rand(1.1, 15) + rand(1.4, 20) + rand(0.9, 15) + rand(3.3, 15) + rand(2.9, 20) + rand(3.8, 15),
        "replicate": [1] * 15 + [2] * 20 + [3] * 15 + [1] * 15 + [2] * 20 + [3] * 15,
    }

    canvas = new_canvas("matplotlib:qt")
    cat_plt = canvas.cat_x(data, x="label", y="value")

    # plot all the raw data
    cat_plt.add_swarmplot(color="replicate", size=8)

    # plot the mean of each replicate
    cat_plt.mean_for_each("replicate").add_markers(
        color="replicate", size=18, symbol="D"
    )

    # plot the mean of all the data for control and treatment
    cat_plt.mean().add_markers(color="black", size=20, symbol="+")

    # plot the mean of replicate means
    cat_plt.mean_for_each("replicate").mean().add_markers(
        color="black", size=30, symbol="_"
    )
    canvas.show(block=True)

if __name__ == "__main__":
    main()
