"""Generate the manuscript-facing mouse-spleen parameter-sensitivity figure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from coreot.results.mouse_spleen_parameter_sensitivity import (
    write_mouse_spleen_parameter_sensitivity,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the mouse-spleen query-penalty sensitivity figure."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    paths = write_mouse_spleen_parameter_sensitivity(
        project_root=args.project_root
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
