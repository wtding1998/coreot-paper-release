"""Load the release validator without importing the repository package facade."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


class ReleaseValidatorLoadError(RuntimeError):
    """Raised when a release does not contain its declared validator."""


def _load_release_module(release_root: str | Path) -> ModuleType:
    release = Path(release_root).resolve()
    source = release / "code/src/coreot/submission/release.py"
    if not source.is_file():
        # Older diagnostic releases predate the sealed-validator bundle.  They
        # remain eligible for current-code compatibility checks, whose
        # authority is explicitly the current checkout.
        from coreot.submission import release as current_release

        return current_release
    module_name = "_coreot_sealed_submission_release"
    existing = sys.modules.get(module_name)
    if existing is not None and Path(str(existing.__file__)).resolve() == source:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ReleaseValidatorLoadError(
            f"Could not construct a module specification for {source}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def validate_submission_release(release_root: str | Path):
    """Validate with the exact validator shipped under ``release_root``."""

    module = _load_release_module(release_root)
    validator = getattr(module, "validate_submission_release", None)
    if validator is None:
        raise ReleaseValidatorLoadError(
            "Release-contained module does not export validate_submission_release"
        )
    return validator(Path(release_root).resolve())
