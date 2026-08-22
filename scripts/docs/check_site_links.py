#!/usr/bin/env python3
"""Check local links, assets, and anchors in a generated TorchLean site."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class PageScan(HTMLParser):
    """Collect a page's base URL, local-link candidates, and named anchors."""

    def __init__(self) -> None:
        super().__init__()
        self.base: str | None = None
        self.ids: set[str] = set()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if identifier := attributes.get("id"):
            self.ids.add(identifier)
        if tag == "base" and (base := attributes.get("href")):
            self.base = base
        if tag in {"a", "link"}:
            if target := attributes.get("href"):
                self.links.append((tag, target))
        elif tag in {"img", "script", "source"}:
            if target := attributes.get("src"):
                self.links.append((tag, target))


def scan_page(path: Path) -> PageScan:
    """Parse one generated HTML page."""

    scan = PageScan()
    scan.feed(path.read_text(encoding="utf-8", errors="replace"))
    return scan


def page_url(site: Path, page: Path) -> str:
    """Give a generated page a synthetic URL for standards-compliant URL resolution."""

    return "http://torchlean.local/" + page.relative_to(site).as_posix()


def local_target(site: Path, url_path: str) -> Path:
    """Map a site-relative URL path to its generated file."""

    target = site / unquote(url_path).lstrip("/")
    if target.is_dir() or (not target.exists() and target.suffix == ""):
        target /= "index.html"
    return target


def check_site(site: Path) -> list[str]:
    """Return every unresolved local target or anchor in ``site``."""

    pages = {page: scan_page(page) for page in site.rglob("*.html")}
    errors: list[str] = []

    for page, scan in list(pages.items()):
        relative_page = page.relative_to(site)
        base = urljoin(page_url(site, page), scan.base or "")
        for tag, raw_target in scan.links:
            if raw_target.startswith(("data:", "javascript:", "mailto:")):
                continue
            resolved = urlsplit(urljoin(base, raw_target))
            if resolved.netloc != "torchlean.local":
                continue
            target = local_target(site, resolved.path)
            try:
                relative_target = target.relative_to(site)
            except ValueError:
                errors.append(f"{relative_page}: `{raw_target}` escapes the generated site")
                continue
            if not target.exists():
                errors.append(
                    f"{relative_page}: missing {tag} target `{raw_target}` "
                    f"(resolved to {relative_target})"
                )
                continue
            if not resolved.fragment or target.suffix != ".html":
                continue
            target_scan = pages.get(target)
            if target_scan is None:
                target_scan = scan_page(target)
                pages[target] = target_scan
            fragment = unquote(resolved.fragment)
            if fragment not in target_scan.ids and fragment != "top":
                errors.append(
                    f"{relative_page}: missing anchor `#{fragment}` in {relative_target}"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="generated site root")
    args = parser.parse_args()
    site = args.site.resolve()
    if not site.is_dir():
        parser.error(f"site directory does not exist: {site}")

    errors = check_site(site)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"FAILED: {len(errors)} unresolved local link(s)")
        return 1
    page_count = sum(1 for _ in site.rglob("*.html"))
    print(f"OK: {page_count} HTML pages have resolvable local links, assets, and anchors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
