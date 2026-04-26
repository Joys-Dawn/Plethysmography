"""One-shot EDF filename normalizer.

Two known issues in the raw data tree, fixed once at the source so the rest of the
pipeline can assume clean filenames:

  1. Trailing whitespace in stem: e.g. ``"250420 4263 p19 .EDF"``.
     Caused by the rig software occasionally appending a space.
  2. Capital-P age token: e.g. ``"260119 5304 P22.EDF"`` should be ``"p22"``.
     Caused by manual entry; the data log uses lowercase.

Both fixes are idempotent; running the script twice is safe (second run is a no-op).
The user authorized in-place rename in the planning conversation.

Run as a script:
    python -m plethysmography.data_loading.filename_normalize --root Data
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple


_AGE_PATTERN = re.compile(r"\bP(\d{2})\b")


def _normalized_name(name: str) -> str:
    """Return the cleaned version of an EDF filename."""
    stem, sep, ext = name.rpartition(".")
    if not sep:
        return name
    cleaned_stem = stem.strip()
    cleaned_stem = _AGE_PATTERN.sub(lambda m: f"p{m.group(1)}", cleaned_stem)
    return f"{cleaned_stem}.{ext}"


def find_dirty_edfs(root: str | Path) -> List[Tuple[Path, str]]:
    """Walk ``root`` and return (path, normalized_name) tuples for any EDF whose
    current name differs from its normalized form."""
    root_path = Path(root)
    seen: set[Path] = set()
    out: List[Tuple[Path, str]] = []
    for pattern in ("*.EDF", "*.edf"):
        for path in root_path.rglob(pattern):
            if path in seen:
                continue
            seen.add(path)
            new = _normalized_name(path.name)
            if new != path.name:
                out.append((path, new))
    return out


def normalize_edf_filenames(
    root: str | Path = "Data",
    dry_run: bool = False,
) -> List[Tuple[Path, Path]]:
    """Rename all dirty EDFs under ``root``. Returns list of (old, new) paths."""
    renames: List[Tuple[Path, Path]] = []
    for path, new_name in find_dirty_edfs(root):
        new_path = path.parent / new_name
        if new_path.exists() and new_path.resolve() != path.resolve():
            print(f"  SKIP (target exists): {path.name} -> {new_name}")
            continue
        print(f"  {path.name!r} -> {new_name!r}")
        if not dry_run:
            path.rename(new_path)
        renames.append((path, new_path))
    if not renames:
        print("All EDF filenames already clean.")
    return renames


def _main() -> None:
    parser = argparse.ArgumentParser(description="Normalize EDF filenames in the Data tree.")
    parser.add_argument("--root", default="Data", help="Directory tree to walk (default: Data)")
    parser.add_argument("--dry-run", action="store_true", help="Print intended renames but do not apply")
    args = parser.parse_args()
    normalize_edf_filenames(args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    _main()
