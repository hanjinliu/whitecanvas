# modified from:
# https://github.com/pyapp-kit/magicgui/blob/main/docs/scripts/_gen_screenshots.py
import re
import warnings
from pathlib import Path
from textwrap import dedent
from typing import Any
from imageio import imwrite
import matplotlib.pyplot as plt
import mkdocs_gen_files

from whitecanvas.canvas import CanvasGrid, SingleCanvas
from whitecanvas.theme import update_default

DOCS: Path = Path(__file__).parent.parent
CODE_BLOCK = re.compile("``` ?python.*?\n([^`]*)```")

def _exec_code(src: str, ns: dict, dest: str) -> dict[str, Any]:
    try:
        exec(src, ns, ns)
    except NameError as e:
        raise NameError(
            f"Error evaluating code for {dest!r} with namespace {set(ns)!r}:\n\n{src}. "
            f"{e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Error evaluating code\n\n{src}\n\nfor {dest!r}"
        ) from e
    return ns

def _write_image(src: str, ns: dict, dest: str) -> None:
    ns = _exec_code(src, ns, dest)
    if "grid" in src and isinstance(ns.get("grid", None), CanvasGrid):
        canvas = ns["grid"]
    elif "canvas" in src:
        canvas = ns["canvas"]
    assert isinstance(canvas, (CanvasGrid, SingleCanvas)), type(canvas)
    with mkdocs_gen_files.open(dest, "wb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()
        img = canvas.screenshot()
        imwrite(f.name, img, format="png")

def _write_html(src: str, ns: dict, dest: str) -> None:
    ns = _exec_code(src, ns, dest)
    if isinstance(ns.get("grid", None), CanvasGrid):
        canvas = ns["grid"]
    else:
        canvas = ns["canvas"]
    assert isinstance(canvas, (CanvasGrid, SingleCanvas)), type(canvas)
    with mkdocs_gen_files.open(dest, "w") as f:
        canvas.to_html(f.name)

EXCLUDE: set[str] = set()


def main() -> None:
    with update_default() as theme:
        theme.canvas_size = (800, 560)
        theme.font.size = 18
    plt.switch_backend("Agg")
    names_found = set[str]()
    for mdfile in sorted(DOCS.rglob("*.md"), reverse=True):
        if mdfile.name in EXCLUDE:
            continue

        md = mdfile.read_text()
        code_blocks = list(CODE_BLOCK.finditer(md))
        namespace = {}
        for match in code_blocks:
            code = dedent(match.group(1)).strip()
            if code.startswith("#!skip"):
                continue
            elif code.startswith("#!name:"):
                if code.endswith(("canvas.show()", "grid.show()")):
                    code = code[:-7]
                line = code.split("\n", 1)[0]
                assert line.startswith("#!name:")
                name = line.split(":", 1)[1].strip()
                if name in names_found:
                    raise ValueError(f"Duplicate name {name!r} in {mdfile}")
                dest = f"_images/{name}.png"
                _write_image(code, namespace, dest)
                names_found.add(f"{name}.png")
            elif code.startswith("#!html:"):
                if code.endswith(("canvas.show()", "grid.show()")):
                    code = code[:-7]
                line = code.split("\n", 1)[0]
                assert line.startswith("#!html:")
                name = line.split(":", 1)[1].strip()
                if name in names_found:
                    raise ValueError(f"Duplicate name {name!r} in {mdfile}")
                dest = f"_images/{name}.html"
                _write_html(code, namespace, dest)
                names_found.add(f"{name}.html")
            else:
                if code.endswith(("canvas.show()", "grid.show()")):
                    code = code[:-7]
                try:
                    exec(code, namespace, namespace)
                except Exception as e:
                    raise RuntimeError(
                        f"Error evaluating code\n\n{code}\n\nfor {dest!r}"
                    ) from e
            # close all if there's more than 10 figures
            if len(plt.get_fignums()) > 10:
                plt.close("all")

main()
