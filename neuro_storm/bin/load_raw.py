#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "tifffile>=2024.1.30",
#     "numpy>=1.26",
# ]
# ///
"""Load raw LBM TIFF data from a file or directory.

Run standalone (uv reads the inline metadata block above)::

    uv run neuro_storm/bin/load_raw.py ~/lbm_data/raw

Or, once the package is installed, via its console-script entry point::

    uvx --from neuro-storm load-raw ~/lbm_data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

TIFF_SUFFIXES = (".tif", ".tiff")


def find_tiffs(path: Path) -> list[Path]:
    """Return TIFF files at ``path`` (the file itself, or sorted files in a dir)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in TIFF_SUFFIXES)
    raise FileNotFoundError(f"No such file or directory: {path}")


def load_raw(path: str | Path) -> list[np.ndarray]:
    """Load every TIFF at ``path`` into a list of arrays, in filename order.

    Args:
        path: A single TIFF file or a directory containing TIFF files.

    Returns:
        One ``numpy.ndarray`` per TIFF file found.
    """
    files = find_tiffs(Path(path).expanduser())
    if not files:
        raise FileNotFoundError(f"No TIFF files found under: {path}")
    return [tifffile.imread(f) for f in files]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load and summarize raw LBM TIFF data.")
    parser.add_argument("path", type=Path, help="TIFF file or directory of TIFF files")
    args = parser.parse_args(argv)

    files = find_tiffs(args.path.expanduser())
    if not files:
        print(f"No TIFF files found under: {args.path}", file=sys.stderr)
        return 1

    for f, arr in zip(files, (tifffile.imread(f) for f in files)):
        print(f"{f.name}\tshape={arr.shape}\tdtype={arr.dtype}")
    print(f"loaded {len(files)} file(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
