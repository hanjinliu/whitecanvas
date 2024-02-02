from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from whitecanvas.utils.normalize import as_array_1d


class Histogram(NamedTuple):
    edges: NDArray[np.number]
    width: float
    counts: NDArray[np.integer]

    def density(self) -> NDArray[np.number]:
        return self.frequency_scaled(self.width)

    def frequency(self) -> NDArray[np.number]:
        return self.frequency_scaled(1)

    def percent(self) -> NDArray[np.number]:
        return self.frequency_scaled(100)

    def scaled(self, scale: float) -> NDArray[np.number]:
        return self.counts / scale

    def frequency_scaled(self, scale: float) -> NDArray[np.number]:
        return self.counts / self.counts.sum() / scale


class HistogramTuple(NamedTuple):
    edges: NDArray[np.number]
    width: float
    counts: list[NDArray[np.integer]]

    def density(self) -> list[NDArray[np.number]]:
        return self.frequency_scaled(self.width)

    def frequency(self) -> list[NDArray[np.number]]:
        return self.frequency_scaled(1)

    def percent(self) -> list[NDArray[np.number]]:
        return self.frequency_scaled(100)

    def scaled(self, scale: float) -> list[NDArray[np.number]]:
        out: list[NDArray[np.number]] = []
        for arr in self.counts:
            scaled = arr / scale
            out.append(scaled)
        return out

    def frequency_scaled(self, scale: float) -> list[NDArray[np.number]]:
        out: list[NDArray[np.number]] = []
        for arr in self.counts:
            density_scaled = arr / arr.sum() / scale
            out.append(density_scaled)
        return out

    def centers(self) -> NDArray[np.number]:
        return (self.edges[:-1] + self.edges[1:]) / 2


def get_hist_edges(
    arrays: list[NDArray[np.number]],
    bins: int | NDArray[np.number],
    range: tuple[float, float] | None = None,
) -> NDArray[np.number]:
    if range is None:
        total = np.concatenate(arrays)
        value_min = total.min()
        value_max = total.max()
    else:
        value_min, value_max = range
        if value_min >= value_max:
            raise ValueError("max must be larger than min in range parameter")
    if isinstance(bins, (int, np.integer)):
        nbins = bins.__index__()
        if nbins < 1:
            raise ValueError("bins should be a positive integer")
        edges = np.linspace(value_min, value_max, nbins + 1)
    else:
        edges = as_array_1d(bins)
        if np.diff(edges).min() <= 0:
            raise ValueError("bin edges must increase monotonically")
    return edges


def histograms(
    arrays: list[NDArray[np.number]],
    bins: int,
    range: tuple[float, float] | None = None,
) -> HistogramTuple:
    edges = get_hist_edges(arrays, bins, range)
    width = edges[1] - edges[0]
    counts = [np.histogram(arr, edges)[0] for arr in arrays]
    return HistogramTuple(edges, width, counts)
