from __future__ import annotations

import os
import re
import warnings
from typing import TYPE_CHECKING, Any


warnings.simplefilter("ignore", DeprecationWarning)

if TYPE_CHECKING:
    from mkdocs.structure.pages import Page


def on_page_markdown(md: str, page: Page, **kwargs: Any) -> str:
    """Called when mkdocs is building the markdown for a page."""

    def _add_images(matchobj: re.Match[str]) -> str:
        code: str = matchobj.group(1).strip()  # source code

        if not code.startswith("#!name:"):
            if code.startswith("#!"):
                _, other = code.split("\n", 1)
            else:
                other = code
            return "```python\n" + other + "\n```"

        code, name = _get_image_name(code)
        code, width = _get_image_width(code)

        reldepth = "../" * page.file.src_path.count(os.sep)
        dest = f"{reldepth}_images/{name}.png"
        link = f"\n![]({dest}){{ loading=lazy, width={width}px }}\n\n"
        new_md = "```python\n" + code + "\n```" + link
        return new_md

    md = re.sub("``` ?python\n([^`]*)```", _add_images, md, flags=re.DOTALL)

    return md

def _get_image_name(code: str) -> tuple[str, str]:
    line, other = code.split("\n", 1)
    assert line.startswith("#!name:")
    name = line.split(":", 1)[1].strip()
    return other, name

def _get_image_width(code: str) -> tuple[str, int]:
    """Get the width of the image from the code."""
    code = code.strip()
    if code.startswith("#!width:"):
        line, other = code.split("\n", 1)
        width = int(line.split(":", 1)[1].strip())
    else:
        other = code
        width = 360
    return other, width
