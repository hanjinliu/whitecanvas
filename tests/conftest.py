import matplotlib.pyplot as plt
import pytest


@pytest.fixture(
    scope="function",
    params=["mock", "matplotlib", "pyqtgraph", "plotly", "bokeh", "vispy"]
)
def backend(request: pytest.FixtureRequest):
    yield request.param
    # TODO: how to skip tests if failed in mock backend?
    if request.param == "matplotlib":
        plt.close("all")
