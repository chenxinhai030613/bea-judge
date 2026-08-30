import os
import shutil
from pathlib import Path


def find_bash() -> str | None:
    candidates: list[Path] = []
    git_executable = shutil.which("git")
    if git_executable:
        git_root = Path(git_executable).resolve().parent.parent
        candidates.append(git_root / "bin" / ("bash.exe" if os.name == "nt" else "bash"))

    bash_executable = shutil.which("bash")
    if bash_executable:
        candidates.append(Path(bash_executable))

    if os.name != "nt":
        candidates.extend([Path("/bin/bash"), Path("/usr/bin/bash")])

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if os.name == "nt" and resolved.name.lower() == "bash.exe" and "system32" in str(resolved).lower():
            continue
        return str(resolved)
    return None
