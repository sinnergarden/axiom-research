#!/usr/bin/env python3
"""Render the stock feature catalog to stdout or an explicit output file."""
import argparse
from pathlib import Path
import sys

# Permit running this checkout's tool without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from axiom_research.feature_catalog import load_feature_catalog, render_feature_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, help="Explicit JSON catalog (default: installed stock_ml_v1)")
    parser.add_argument("--output", type=Path, help="Explicit Markdown destination; default is stdout")
    args = parser.parse_args()
    text = render_feature_catalog(load_feature_catalog(args.catalog))
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

