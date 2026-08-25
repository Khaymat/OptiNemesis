"""Architecture guard: enforces the dependency DAG from docs/ARCHITECTURE.md.

Layers (lower may never be imported by higher-numbered layers... i.e., imports
may only point toward lower layers):

    0 core
    1 stats
    2 families
    3 runners
    4 search / validate / diagnostics
    5 reporting

Adapters are special: they may import ONLY ``core`` inside the package.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "optinemesis"

LAYER_OF = {
    "core": 0,
    "stats": 1,
    "families": 2,
    "runners": 3,
    "search": 4,
    "validate": 4,
    "diagnostics": 4,
    "reporting": 5,
}


def _module_layer(module: str) -> int | None:
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "optinemesis":
        return None
    top = parts[1]
    if top == "adapters":
        return -1
    return LAYER_OF.get(top)


def _iter_python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return {m for m in modules if m.startswith("optinemesis")}


def _layer_violations() -> list[str]:
    violations: list[str] = []
    for path in _iter_python_files():
        rel = path.relative_to(SRC).with_suffix("")
        parts = rel.parts
        own_module = ".".join(("optinemesis", *parts))
        own_top = parts[0]
        own_layer = LAYER_OF.get(own_top)
        if own_layer is None:
            continue
        is_adapter = own_top == "adapters"
        for imported in sorted(_imports_of(path)):
            imported_layer = _module_layer(imported)
            if imported_layer is None and not imported.startswith("optinemesis.adapters"):
                continue
            imported_top = imported.split(".")[1]
            if is_adapter:
                if imported_top != "core":
                    violations.append(f"{own_module} imports {imported} (adapters: core only)")
                continue
            target_layer = LAYER_OF.get(imported_top)
            if target_layer is not None and target_layer > own_layer:
                violations.append(
                    f"{own_module} (layer {own_layer}) imports {imported} "
                    f"(layer {target_layer})"
                )
    return violations


def test_dependency_dag_respected() -> None:
    violations = _layer_violations()
    assert not violations, "\n".join(violations)

