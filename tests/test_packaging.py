"""Packaging regression guard (R7.7): every module in the installed `umwelt`
package must import cleanly.

The class of bug this guards: a module inside the package importing something
that only exists in the repo checkout (the kits' old `from examples...`
imports could never resolve from an installed wheel — found and fixed by the
2026-07-31 kits→examples/kits move). If a module quietly grows a dependency
on repo-layout context, this test fails at the exact module, not at some
downstream consumer's boot.
"""
from __future__ import annotations

import importlib
import pkgutil

import umwelt


def test_every_umwelt_module_imports():
    failures = []
    for mod in pkgutil.walk_packages(umwelt.__path__, prefix="umwelt."):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001 — report every failure mode
            failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules that fail to import:\n" + "\n".join(failures)
