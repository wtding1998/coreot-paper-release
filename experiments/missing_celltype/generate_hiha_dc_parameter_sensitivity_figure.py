"""Generate the manuscript-facing HIHA query-penalty sensitivity figure."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coreot.results.hiha_parameter_sensitivity import (  # noqa: E402
    write_hiha_parameter_sensitivity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the HIHA query-penalty sensitivity source bundle."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_hiha_parameter_sensitivity(project_root=args.project_root)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
