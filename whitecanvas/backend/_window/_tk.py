from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

import numpy as np
from PIL import Image, ImageTk
from psygnal import Signal

from whitecanvas._axis import CategoricalAxis, DimAxis, RangeAxis
from whitecanvas.canvas import CanvasBase, CanvasGrid


class TkCanvas(ttk.Frame):
    def __init__(self, canvas: CanvasGrid):
        super().__init__()
        tkc = tk.Canvas(width=300, height=300)
        tkc.pack()
        self._tkcanvas = tkc
        self._canvas = canvas
        self._canvas_qimage = ImageTk.PhotoImage(
            image=Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8))
        )

    def resize(self, event: tk.Event):
        if event.widget.widgetName == "toplevel":
            w, h = event.width, event.height
            self._canvas.size = w, h
            self._update_imagetk()
            self.update()

    def _update_imagetk(self):
        buf = self._canvas.screenshot()
        self._canvas_qimage = ImageTk.PhotoImage(image=Image.fromarray(buf))
        self._tkcanvas.create_image(20, 20, anchor="nw", image=self._canvas_qimage)


class TkMainWindow(ttk.Frame):
    _instance = None

    def __init__(self, canvas: CanvasGrid):
        super().__init__()
        tkcanvas = TkCanvas(canvas)
        tkcanvas.pack(fill="both", expand=True)
        sl = TkDimSliders(self)
        sl.pack(fill="both", expand=True)
        self.__class__._instance = self
        canvas.events.drawn.connect(tkcanvas._update_imagetk, max_args=0)


class TkSlider(ttk.Scale):
    changed = Signal(int)

    def __init__(self, parent=None, **kwargs):
        kwargs["command"] = lambda value: self.changed.emit(int(value))
        super().__init__(parent, **kwargs)


class TkComboBox(ttk.Combobox):
    changed = Signal(str)

    def __init__(self, parent=None, **kwargs):
        kwargs["postcommand"] = lambda: self.changed.emit(self.get())
        super().__init__(parent, **kwargs)


class TkDimSliders(ttk.Frame):
    changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = ttk.Frame(self)
        self._layout.pack(fill="both", expand=True)
        self._widgets = {}

    def set_axes(self, axes: list[DimAxis]):
        for axis in axes:
            if isinstance(axis, RangeAxis):
                slider = TkSlider(
                    self._layout,
                    from_=0,
                    to=axis.size(),
                    value=axis.value(),
                    orient="horizontal",
                )
                slider.pack(fill="both", expand=True)
            elif isinstance(axis, CategoricalAxis):
                slider = TkComboBox(
                    self._layout,
                    values=axis.categories(),
                    # state="readonly",
                    textvariable=axis.value(),
                )
                slider.pack(fill="both", expand=True)
            else:
                raise ValueError(f"Unknown axis type {axis}")
            self._widgets[axis.name] = slider

    def connect_changed(self, callback: Callable[[dict[str, object]], None]):
        self.changed.connect(callback, max_args=1)

    def _emit_changed(self):
        values = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, TkSlider):
                values[name] = int(widget.get())
            elif isinstance(widget, TkComboBox):
                values[name] = widget.get()
            else:
                raise NotImplementedError
        self.changed.emit(values)

    def show(self):
        self.mainloop()

    @classmethod
    def from_canvas(cls, canvas: CanvasBase, parent=None):
        self = cls(parent=parent)

        @canvas.dims.events.axis_names.connect
        def _update_axes():
            self.set_axes(canvas.dims._axes)

        self.connect_changed(canvas.dims.set_indices)
        _update_axes()
        return self
