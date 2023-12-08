from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Iterable, TypeVar, Generic
import numpy as np
from numpy.typing import NDArray
from whitecanvas._exceptions import ReferenceDeletedError

if TYPE_CHECKING:
    from whitecanvas.canvas import CanvasGrid

_T = TypeVar("_T")


class Animation:
    """
    Class for capturing animations.

    You can either manually capture frames using the `capture` method, or use
    this instance as a iterator: `anim(iterable)` or `anim.iter_range(num)`.

    Examples
    --------
    >>> from whitecanvas import plot as plt
    >>> from whitecanvas.animation import Animation
    >>> import numpy as np
    >>> ### Create a canvas and a line
    >>> canvas = plt.figure()
    >>> x = np.linspace(0, 4 * np.pi, 200)
    >>> line = plt.line(x, np.sin(x))
    >>> ### Create an animation
    >>> anim = Animation(canvas)
    >>> for i in anim.iter_range(20):
    ...     line.set_data(x, np.sin(x + i * np.pi / 10))
    >>> ### Save the animation
    >>> anim.save("animation.gif")
    """

    def __init__(self, grid: CanvasGrid):
        self._grid_ref = weakref.ref(grid)
        self._frames: list[NDArray[np.int8]] = []

    def __call__(self, iterable: Iterable[_T]) -> AnimationIterator[_T]:
        return AnimationIterator(self, iterable)

    def iter_range(self, *args) -> AnimationIterator[int]:
        """Equivalent to `anim(range(*args))`."""
        return AnimationIterator(self, range(*args))

    def _grid(self) -> CanvasGrid:
        grid = self._grid_ref()
        if grid is None:
            raise ReferenceDeletedError("CanvasGrid has been deleted.")
        return grid

    def capture(self):
        """Capture current canvas state."""
        grid = self._grid()
        self._frames.append(grid.screenshot())

    def save(self, filename: str, dt: float = 100, loop: int = 0):
        """Save animation to a file."""
        import imageio

        imageio.mimwrite(filename, self._frames, duration=dt, loop=loop)

    def asarray(self) -> NDArray[np.int8]:
        """Convert frames to a (N, X, Y, 4) numpy array."""
        return np.stack(self._frames, axis=0)


class AnimationIterator(Generic[_T]):
    def __init__(self, anim: Animation, iterable: Iterable[_T]):
        self._anim = anim
        self._iterable = iterable

    def __iter__(self):
        grid = self._anim._grid()
        for item in self._iterable:
            self._anim._frames.append(grid.screenshot())
            yield item
        self._anim._frames.append(grid.screenshot())
