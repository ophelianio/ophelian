#!/usr/bin/env python3
"""Verify that every `#fragment` markdown link points at a real heading.

Companion to the lychee-based link checker in
`.github/workflows/links.yml`. We do anchor verification ourselves
(instead of via `lychee --include-fragments`) because lychee's
heading-slug algorithm differs from GitHub Flavored Markdown for
headings containing `&`, which would false-positive against the
README's TOC entries like `#status--versioning`.

Algorithm matches GFM (https://github.github.com/gfm/) closely
enough for the docs in this repo:

    1. lowercase the heading text
    2. strip backticks, asterisks, underscores (markdown emphasis)
    3. remove all non-(letter | digit | space | dash | underscore)
    4. replace each whitespace character with a single dash;
       existing dashes are preserved (so `Status & versioning`
       becomes `status--versioning`, not `status-versioning`)

Limitations (all currently safe for this repo):

    - We do NOT implement GitHub's `-1`, `-2` suffix de-duplication
      for repeated headings. None of the scanned files have repeated
      headings today; if that changes, add a check here.
    - HTML headings inside markdown (`<h2>foo</h2>`) are ignored.
    - Footnote anchors (`[^1]`) are ignored — different namespace.

Exit codes:
    0 — every #fragment link resolves
    1 — at least one broken anchor (printed with file:line)
    2 — internal error (e.g. a glob matched nothing)
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

# --- markdown parsing --------------------------------------------------

# `[text](target)` and `![alt](target)`. We don't need to distinguish
# them because we only care about the target's #fragment.
LINK_RE = re.compile(r"!?\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)\)")

# Markdown ATX headings: `# Foo`, `## Bar`, ... up to `######`.
# Setext headings (===, ---) are not used in this repo.
HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*#*\s*$")


def gfm_slug(text: str) -> str:
    """GFM-compatible heading slug.

    Behavior is documented in the module docstring; this function
    must stay in sync with it.
    """
    s = text.lower().strip()
    # strip markdown emphasis and inline code markers
    s = re.sub(r"[`*_]", "", s)
    # remove anything not letter/digit/space/dash/underscore
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    # any whitespace char → single dash; keep consecutive
    s = re.sub(r"\s", "-", s)
    return s


def collect_headings(md_path: Path) -> set[str]:
    """All GFM-slugged anchors defined by `# Heading` lines in a file."""
    slugs: set[str] = set()
    in_fence = False
    for line in md_path.read_text(encoding="utf-8").splitlines():
        # Skip lines inside fenced code blocks (```...```).
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if m:
            slugs.add(gfm_slug(m.group("text")))
    return slugs


def collect_anchor_links(md_path: Path) -> list[tuple[int, str, str]]:
    """Yield (line_number, target_file_or_self, fragment) for each
    markdown link whose target contains a `#fragment`.

    `target_file_or_self` is empty when the link is purely to an
    anchor in the same file (e.g. `[TOC](#install)`).
    """
    out: list[tuple[int, str, str]] = []
    in_fence = False
    for lineno, line in enumerate(
        md_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in LINK_RE.finditer(line):
            target = m.group("target")
            # External URLs and protocol links: not our problem here.
            if target.startswith(("http://", "https://", "mailto:", "ftp:", "tel:")):
                continue
            if "#" not in target:
                continue
            file_part, _, frag = target.partition("#")
            if not frag:
                continue
            out.append((lineno, file_part, frag))
    return out


# --- driver ------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        default=[
            "README.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            "CHANGELOG.md",
            "SECURITY.md",
        ],
        help="Markdown files (and globs) to scan. Defaults to the "
        "repo's top-level policy files plus README.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=["docs/**/*.md"],
        help="Glob(s) to expand and add to the file list. Repeatable.",
    )
    args = parser.parse_args(argv)

    # Resolve all files (literal + glob expansion). Preserve order;
    # de-duplicate with a set.
    seen: set[str] = set()
    files: list[Path] = []
    for spec in args.files:
        for matched in sorted(glob.glob(spec, recursive=True)) or [spec]:
            if matched not in seen and Path(matched).is_file():
                files.append(Path(matched))
                seen.add(matched)
    for spec in args.glob:
        for matched in sorted(glob.glob(spec, recursive=True)):
            if matched not in seen and Path(matched).is_file():
                files.append(Path(matched))
                seen.add(matched)

    if not files:
        print("error: no files matched", file=sys.stderr)
        return 2

    # Pre-compute the heading set per file so cross-file anchor
    # lookups (e.g. `[setup](CONTRIBUTING.md#dev-setup)`) work.
    headings_by_file: dict[Path, set[str]] = {f: collect_headings(f) for f in files}

    errors: list[str] = []
    total_anchors = 0
    for f in files:
        for lineno, file_part, frag in collect_anchor_links(f):
            total_anchors += 1
            target_file = f if file_part == "" else (f.parent / file_part).resolve()
            target_file = Path(target_file)
            if target_file not in headings_by_file:
                # Anchor target points at a file outside our scan
                # surface (e.g. an examples/*.py). Skip — internal
                # job already verified the file exists.
                if target_file.exists() and target_file.suffix == ".md":
                    headings_by_file[target_file] = collect_headings(target_file)
                else:
                    continue
            if frag not in headings_by_file[target_file]:
                target_label = (
                    "self" if file_part == "" else str(target_file.relative_to(Path.cwd()))
                )
                errors.append(
                    f"  {f}:{lineno}: #{frag}  →  no such heading in {target_label}"
                )

    if errors:
        print(f"\nBroken anchors ({len(errors)} of {total_anchors}):", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return 1

    print(f"OK: {total_anchors} anchor link(s) verified across {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
