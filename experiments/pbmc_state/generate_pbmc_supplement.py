from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coreot.results.pbmc_supplement import generate_pbmc_supplement  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PBMC supplementary source data, figures, and their "
            "provenance manifest without editing manuscript prose."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root.",
    )
    args = parser.parse_args()
    outputs = generate_pbmc_supplement(args.project_root)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
