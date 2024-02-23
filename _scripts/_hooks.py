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
        prefix = matchobj.group(0).split("\n", 1)[0]  # ``` python ...`
        code: str = matchobj.group(1).strip()  # source code

        if code.startswith("#!name:"):
            code, name = _get_image_name(code)
            code, width = _get_image_width(code)

            reldepth = "../" * page.file.src_path.count(os.sep)
            dest = f"{reldepth}_images/{name}.png"
            link = f"\n![]({dest}){{ loading=lazy, width={width}px }}\n\n"
            new_md = f"{prefix}\n{code}\n```{link}"
            return new_md
        elif code.startswith("#!html:"):
            code, name = _get_html_name(code)
            reldepth = "../" * (page.file.src_path.count(os.sep) + 1)
            dest = f"{reldepth}_images/{name}.html"
            html_text = (
                f'<iframe src={dest} frameborder="0" width="400px" height="300px" '
                'scrolling="no"></iframe>'
            )
            new_md = f"{prefix}\n{code}\n```\n\n{html_text}\n"
            return new_md
        elif code.startswith("#!"):
            _, other = code.split("\n", 1)
        else:
            other = code
        return f"{prefix}\n{other}\n```"


    # md = re.sub("``` ?python\n([^`]*)```", _add_images, md, flags=re.DOTALL)
    md = re.sub("``` ?python.*?\n([^`]*)```", _add_images, md)

    return md

def _get_image_name(code: str) -> tuple[str, str]:
    line, other = code.split("\n", 1)
    assert line.startswith("#!name:")
    name = line.split(":", 1)[1].strip()
    return other, name

def _get_html_name(code: str) -> tuple[str, str]:
    line, other = code.split("\n", 1)
    assert line.startswith("#!html:")
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
