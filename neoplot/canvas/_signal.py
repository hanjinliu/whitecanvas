from __future__ import annotations
from typing import Callable, Generic, TypeVar, Generator
import inspect
import warnings
from psygnal import Signal

_R = TypeVar("_R")
_S = TypeVar("_S")
_Y = TypeVar("_Y")

GeneratorFunction = Callable[..., Generator[_R, _S, _Y]]
_G = TypeVar("_G", bound=GeneratorFunction)


class _Slot(Generic[_G]):
    _NOT_STARTED = object()
    _FINISHED = object()

    def __init__(self, slot: _G):
        self._slot = slot
        self._generator: Generator | object = self._NOT_STARTED

    def init(self):
        """Initialize the slot."""
        self._generator = self._NOT_STARTED

    def next(self, *args):
        """Advance the slot."""
        if self._generator is self._FINISHED:
            return
        if self._generator is self._NOT_STARTED:
            self._generator = self._slot(*args)
        try:
            next(self._generator)
        except StopIteration:
            self._generator = self._FINISHED
            return

    def terminate(self, *args, warn: bool = True):
        """Terminate the slot."""
        self.next(*args)
        if self._generator is self._FINISHED:
            return
        if warn:
            warnings.warn(
                f"Generator {self._slot!r} did not terminate.",
                UserWarning,
                stacklevel=2,
            )


class GeneratorSignal:
    def __init__(
        self,
    ):
        self._slots: list[_Slot] = []

    def connect(self, slot: _G | None = None) -> _G:
        if not callable(slot):
            raise TypeError(f"Can only connect callable object, got {slot!r}")
        if not inspect.isgeneratorfunction(slot):
            raise TypeError(f"Can only connect generator function, got {slot!r}")
        self._slots.append(_Slot(slot))
        return slot

    def disconnect(self, slot: GeneratorFunction | None = None, missing_ok: bool = True) -> None:
        if slot is None:
            return self._slots.clear()
        i = -1
        for _i, _slot in enumerate(self._slots):
            if _slot._slot is slot:
                i = _i
                break
        if i > 0:
            self._slots.pop(i)
        elif not missing_ok:
            raise ValueError(f"Slot {slot!r} not found")
        return

    def emit(self, *args) -> None:
        for slot in self._slots:
            slot.next(*args)
