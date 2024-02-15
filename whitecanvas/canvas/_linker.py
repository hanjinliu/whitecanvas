from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar
from weakref import WeakSet

if TYPE_CHECKING:
    from whitecanvas.canvas._namespaces import AxisNamespace


class AxisLinker:
    _GLOBAL_LINKERS: ClassVar[set[AxisLinker]] = set()

    def __new__(cls):
        self = super().__new__(cls)
        cls._GLOBAL_LINKERS.add(self)
        return self

    def __init__(self):
        self._axis_set = WeakSet["AxisNamespace"]()
        self._updating = False

    def link(self, axis: AxisNamespace):
        """Link an axis."""
        axis._get_canvas()  # raise error if the parent canvas is deleted.
        if axis in self._axis_set:
            warnings.warn(f"Axis {axis} already linked", RuntimeWarning, stacklevel=2)
            return
        self._axis_set.add(axis)
        axis.events.lim.connect(self.set_limits)

    def unlink(self, axis: AxisNamespace):
        """Unlink an axis."""
        if axis not in self._axis_set:
            warnings.warn(f"Axis {axis} was not linked", RuntimeWarning, stacklevel=2)
        self._axis_set.discard(axis)
        axis.events.lim.disconnect(self.set_limits)

    def unlink_all(self) -> None:
        """Unlink all axes."""
        for axis in self._axis_set:
            self.unlink(axis)
        self.__class__._GLOBAL_LINKERS.discard(self)

    def is_alive(self) -> bool:
        """Check if the linker is still alive."""
        return self in self.__class__._GLOBAL_LINKERS

    def set_limits(self, limits: tuple[float, float]):
        if self._updating:
            return
        self._updating = True
        try:
            for axis in self._axis_set:
                axis.lim = limits
        finally:
            self._updating = False

    @classmethod
    def link_axes(cls, *axes: AxisNamespace):
        """Link multiple axes."""
        self = cls()
        if len(axes) == 1 and hasattr(axes[0], "__iter__"):
            axes = axes[0]
        for axis in axes:
            self.link(axis)
        return self


class AxisLinkerRef:
    def __init__(self, linker: AxisLinker):
        self._linker = linker

    def _get_linker(self):
        if self._linker.is_alive():
            return self._linker
        raise RuntimeError("Linker has been deleted")

    def link(self, axis: AxisNamespace):
        """Link an axis."""
        self._get_linker().link(axis)
        return self

    def unlink(self, axis: AxisNamespace):
        """Unlink an axis."""
        self._get_linker().unlink(axis)
        return self

    def unlink_all(self) -> None:
        """Unlink all axes."""
        self._get_linker().unlink_all()


def link_axes(*axes: AxisNamespace):
    """Link multiple axes."""
    linker = AxisLinker.link_axes(*axes)
    return AxisLinkerRef(linker)
