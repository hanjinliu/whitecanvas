from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from whitecanvas.types import HistBinType


class HistogramTuple(NamedTuple):
    edges: NDArray[np.number]
    counts: list[NDArray[np.integer]]

    def density(self) -> list[NDArray[np.number]]:
        return self.probability_scaled(self.widths())

    def probability(self) -> list[NDArray[np.number]]:
        return self.probability_scaled(1)

    def percent(self) -> list[NDArray[np.number]]:
        return self.probability_scaled(100)

    def frequency(self) -> list[NDArray[np.number]]:
        return self.scaled(self.widths())

    def scaled(self, scale: float | NDArray[np.number]) -> list[NDArray[np.number]]:
        out: list[NDArray[np.number]] = []
        for arr in self.counts:
            scaled = arr / scale
            out.append(scaled)
        return out

    def probability_scaled(
        self,
        scale: float | NDArray[np.number],
    ) -> list[NDArray[np.number]]:
        out: list[NDArray[np.number]] = []
        for arr in self.counts:
            density_scaled = arr / arr.sum() / scale
            out.append(density_scaled)
        return out

    def centers(self) -> NDArray[np.number]:
        return (self.edges[:-1] + self.edges[1:]) / 2

    def widths(self) -> NDArray[np.number]:
        return np.diff(self.edges)


def get_hist_edges(
    arrays: list[NDArray[np.number]],
    bins: HistBinType,
    range: tuple[float, float] | None = None,
) -> NDArray[np.number]:
    return np.histogram_bin_edges(np.concatenate(arrays), bins, range)


def histograms(
    arrays: list[NDArray[np.number]],
    bins: HistBinType,
    range: tuple[float, float] | None = None,
) -> HistogramTuple:
    edges = get_hist_edges(arrays, bins, range)
    counts = [np.histogram(arr, edges)[0] for arr in arrays]
    return HistogramTuple(edges, counts)
