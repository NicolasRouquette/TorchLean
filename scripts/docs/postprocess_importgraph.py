#!/usr/bin/env python3
"""Apply TorchLean site integration fixes to generated import-graph HTML."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MATHLIB_DOCS_URL = (
    'var docs_url = params.get("docs_url") || '
    '"https://leanprover-community.github.io/mathlib4_docs/";'
)
TORCHLEAN_DOCS_URL = (
    'var docs_url = params.get("docs_url") || '
    'new URL("../docs/", window.location.href).href;'
)
LABEL_THRESHOLD = re.compile(
    r'(?:    // [^\n]*\n)?    labelRenderedSizeThreshold: preferDark \? 1000000000 : 9,'
)
LABEL_THRESHOLD_REPLACEMENT = (
    "    // Sigma's label color is fixed in this generated viewer; dark mode favors graph "
    "structure over labels.\n"
    "    labelRenderedSizeThreshold: preferDark ? 1000000000 : 9,"
)


def postprocess(page: Path) -> None:
    html = page.read_text(encoding="utf-8")
    html = html.replace('  <link rel="stylesheet" href="style.css" />\n', "")
    html = html.replace(MATHLIB_DOCS_URL, TORCHLEAN_DOCS_URL)
    html = LABEL_THRESHOLD.sub(LABEL_THRESHOLD_REPLACEMENT, html, count=1)
    page.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("page", type=Path, help="generated importgraph index.html")
    args = parser.parse_args()
    postprocess(args.page)


if __name__ == "__main__":
    main()
