from __future__ import annotations

import ast
from pathlib import Path

import mini_deepagents_demo

FORBIDDEN = (
    "fastapi",
    "starlette",
    "httpx",
    "sqlite3",
    "uvicorn",
    "pydantic",
    "logging",
    "mini_deepagents_demo.adapters",
    "mini_deepagents_demo.main",
)


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def _is_forbidden(module: str) -> bool:
    return any(module == name or module.startswith(f"{name}.") for name in FORBIDDEN)


def test_domain_core_imports_no_framework_or_adapter() -> None:
    package = Path(mini_deepagents_demo.__file__).parent
    files = [*sorted((package / "chat").rglob("*.py")), package / "errors.py"]

    violations = sorted(
        f"{path.relative_to(package)}: {module}"
        for path in files
        for module in _imported_modules(path)
        if _is_forbidden(module)
    )

    assert violations == []
