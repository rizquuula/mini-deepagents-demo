from __future__ import annotations

import ast
import pkgutil
from pathlib import Path

import mini_deepagents

FRAMEWORK = "mini_deepagents"
FRAMEWORK_SUBMODULES = {
    f"{FRAMEWORK}.{info.name}" for info in pkgutil.iter_modules(mini_deepagents.__path__)
}

ROOT = Path(__file__).resolve().parents[2]
DEMO_PACKAGE = ROOT / "src" / "backend" / "mini_deepagents_demo"
TESTS = ROOT / "tests" / "backend"


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
            modules.update(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if f"{node.module}.{alias.name}" in FRAMEWORK_SUBMODULES
            )
    return modules


def _is_framework(module: str) -> bool:
    return module == FRAMEWORK or module.startswith(f"{FRAMEWORK}.")


def _framework_imports_outside(folder: Path, allowed: set[str]) -> list[str]:
    return sorted(
        f"{path.relative_to(ROOT)}: {module}"
        for path in sorted(folder.rglob("*.py"))
        for module in _imported_modules(path)
        if _is_framework(module) and module not in allowed
    )


def test_demo_has_no_domain_core_of_its_own() -> None:
    assert not (DEMO_PACKAGE / "chat").exists()


def test_production_code_imports_only_the_framework_top_level() -> None:
    assert _framework_imports_outside(DEMO_PACKAGE, {FRAMEWORK}) == []


def test_test_code_imports_only_the_framework_top_level_and_testing() -> None:
    allowed = {FRAMEWORK, f"{FRAMEWORK}.testing"}
    assert _framework_imports_outside(TESTS, allowed) == []
