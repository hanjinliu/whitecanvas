import matplotlib.pyplot as plt
import pytest

ALL_BACKENDS = ["mock", "matplotlib", "pyqtgraph", "plotly", "bokeh", "vispy"]

def pytest_addoption(parser):
    parser.addoption("--backend", default="all")

@pytest.fixture(scope="function", params=ALL_BACKENDS)
def backend(request: pytest.FixtureRequest):
    backend = request.config.getoption("--backend")
    if backend != "all":
        if isinstance(backend, str):
            request.param = backend
        else:
            backend = list(backend)
            request.param.param = backend
    yield request.param
    # TODO: how to skip tests if failed in mock backend?
    if request.param == "matplotlib":
        plt.close("all")
