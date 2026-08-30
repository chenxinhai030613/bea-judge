from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_project_path(root: Path, value: Any) -> Path:
    raw = str(value).replace("\\", "/")
    path = Path(raw)
    return path if path.is_absolute() else root / path
