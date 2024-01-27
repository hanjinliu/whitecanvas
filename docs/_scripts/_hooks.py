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

        line, other = code.split("\n", 1)
        assert line.startswith("#!name:")
        name = line.split(":", 1)[1].strip()
        dest = f"_images/{name}.png"

        reldepth = "../" * page.file.src_path.count(os.sep)
        dest = f"{reldepth}_images/{name}.png"
        link = f"\n![]({dest}){{ loading=lazy, width=360px }}\n\n"
        new_md = "```python\n" + other + "\n```" + link
        return new_md

    md = re.sub("``` ?python\n([^`]*)```", _add_images, md, re.DOTALL)

    return md
