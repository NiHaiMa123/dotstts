from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


def validate_output_path(
    output: str | Path, *, protected_inputs: Iterable[str | Path] = ()
) -> Path:
    """Validate resolved paths before any mkdir, recovery, or output write."""
    resolved = Path(output).resolve()
    for raw in (REPO_ROOT / "data/inbox", *protected_inputs):
        protected = Path(raw).resolve()
        if (
            resolved == protected
            or protected in resolved.parents
            or resolved in protected.parents
        ):
            raise ValueError(f"long-form output overlaps protected input: {resolved} / {protected}")
    return resolved


def contained_file(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or PureWindowsPath(relative).drive:
        raise ValueError("audio path must be relative to its manifest")
    resolved = (root / relative).resolve()
    resolved.relative_to(root.resolve())
    return resolved
