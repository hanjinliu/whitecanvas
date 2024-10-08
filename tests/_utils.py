from contextlib import contextmanager
import warnings
from cmap import Color

def assert_color_equal(a, b):
    if Color(a) != Color(b):
        raise AssertionError(f"Color {a} != {b}")

def assert_color_array_equal(arr, b):
    cols = [Color(a) for a in arr]
    try:
        other = Color(b)
    except ValueError:
        other = [Color(each) for each in b]
    if isinstance(other, Color):
        ok = all([each == other for each in cols])
    else:
        if len(cols) != len(other):
            ok = False
        else:
            ok = all([a == b for a, b in zip(cols, other)])
    if not ok:
        raise AssertionError(f"Color {arr} != {b}")

@contextmanager
def filter_warning(backend: str, choices: "str | list[str]"):
    if isinstance(choices, str):
        choices = [choices]
    if backend in choices:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    else:
        yield
