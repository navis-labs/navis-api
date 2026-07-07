"""Architecture enforcement: platform must contain no business-domain coupling.

Rule 2/9 of the constitution: platform is shared infrastructure only. It must
never import a business domain — otherwise infrastructure becomes a hidden
dependency hub and the dependency graph inverts. This test fails the build if
any `navis/platform/**` module imports a business domain.
"""
from __future__ import annotations

import ast
import pathlib

PLATFORM_DIR = pathlib.Path(__file__).resolve().parent.parent / "navis" / "platform"

BUSINESS_DOMAINS = {
    "identity",
    "identity_resolution",
    "signals",
    "memory",
    "state",
    "reasoning",
    "decisions",
    "execution",
    "observation",
    "learning",
    "intelligence",
    "ml",
    "orchestration",
    "api",
}


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_platform_does_not_import_business_domains():
    violations: list[str] = []
    for py in PLATFORM_DIR.rglob("*.py"):
        for mod in _imported_modules(py):
            parts = mod.split(".")
            if len(parts) >= 2 and parts[0] == "navis" and parts[1] in BUSINESS_DOMAINS:
                rel = py.relative_to(PLATFORM_DIR.parent.parent)
                violations.append(f"{rel} imports {mod}")
    assert not violations, "platform leaks into business domains:\n" + "\n".join(violations)


def test_platform_modules_all_import_cleanly():
    """Importing every platform module must not raise (no broken wiring)."""
    import importlib

    modules = [
        "navis.platform.config",
        "navis.platform.database",
        "navis.platform.cache",
        "navis.platform.queue",
        "navis.platform.events",
        "navis.platform.security",
        "navis.platform.monitoring",
        "navis.platform.monitoring.health",
        "navis.platform.monitoring.middleware",
        "navis.platform.monitoring.sentry",
        "navis.platform.storage",
    ]
    for m in modules:
        importlib.import_module(m)
