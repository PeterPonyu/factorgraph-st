from __future__ import annotations

import pathlib
try:
    import tomllib  # Python >=3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # Python 3.10


def test_numpy_is_runtime_dependency_for_numpy_importing_modules():
    root = pathlib.Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    deps = pyproject["project"].get("dependencies", [])
    assert any(dep.lower().startswith("numpy") for dep in deps)


def test_test_extra_does_not_hide_runtime_numpy_requirement():
    root = pathlib.Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    test_deps = pyproject["project"].get("optional-dependencies", {}).get("test", [])
    assert all(not dep.lower().startswith("numpy") for dep in test_deps)
