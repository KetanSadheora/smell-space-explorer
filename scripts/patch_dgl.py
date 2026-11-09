"""Idempotently install the OpenPOM DGL GraphBolt stub.

Some DGL/PyTorch combinations fail while importing DGL's optional GraphBolt
extension. OpenPOM does not use GraphBolt, so this script replaces
``dgl/graphbolt/__init__.py`` with a minimal stub.

The script deliberately does **not** import dgl: the failing import would stop
execution before the file could be patched. It locates the package using
``sys.prefix`` and ``site.getsitepackages()`` instead.
"""

from __future__ import annotations

import site
import sys
from pathlib import Path

STUB = (
    '"""DGL Graphbolt stub for OpenPOM."""\n'
    "def load_graphbolt():\n"
    "    pass\n\n"
    "def __getattr__(name):\n"
    "    return None\n"
)


def candidate_paths() -> list[Path]:
    """Return possible ``dgl/graphbolt/__init__.py`` paths for this interpreter."""
    roots = [Path(sys.prefix), Path(sys.prefix) / "Lib", Path(sys.prefix) / "lib"]
    try:
        roots.extend(Path(path) for path in site.getsitepackages())
    except AttributeError:
        pass
    try:
        roots.append(Path(site.getusersitepackages()))
    except AttributeError:
        pass

    candidates: list[Path] = []
    for root in roots:
        for site_packages in (root, root / "site-packages", root / "Lib" / "site-packages"):
            candidate = site_packages / "dgl" / "graphbolt" / "__init__.py"
            if candidate.exists() and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def patch_graphbolt(path: Path) -> bool:
    """Write the stub to ``path``.

    Returns:
        ``True`` if the file was changed, ``False`` if it was already stubbed.
    """
    if path.read_text(encoding="utf-8") == STUB:
        return False
    path.write_text(STUB, encoding="utf-8")
    return True


def main() -> None:
    """Locate and patch the DGL GraphBolt package for the current environment."""
    candidates = candidate_paths()
    if not candidates:
        raise FileNotFoundError(
            "Could not find dgl/graphbolt/__init__.py under "
            f"{sys.prefix!r} or site.getsitepackages(). Install DGL first."
        )

    for path in candidates:
        changed = patch_graphbolt(path)
        status = "patched" if changed else "already patched"
        print(f"{status}: {path}")


if __name__ == "__main__":
    main()
