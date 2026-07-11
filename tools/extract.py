#!/usr/bin/env python3
"""Curated-copy extraction from the meerkat tree (read-only source) into umwelt.

Mechanical half of the port: copy each manifest file, rewrite imports through the
module map, rename env-var prefixes, and report every import it could NOT resolve —
that report is the hand-curation worklist. Hand-edited files are listed in FROZEN
once curated; re-runs skip them so mechanical refreshes never clobber hand work.

Usage: python3 tools/extract.py [--phase p1] [--force path ...]
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

MEERKAT = Path(os.environ.get("MEERKAT_ROOT", "~/ws/smrthaus/meerkat-dev")).expanduser()
REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "umwelt"
BRAIN = "meerkat/brain"

# meerkat module name (relative to meerkat/brain) -> umwelt dotted module.
# Includes forward declarations for later-phase targets so lazy imports rewrite
# to their FINAL homes even before those files exist (lazy = safe until called).
MAP = {
    # substrate
    "bloch": "substrate.bloch", "density_matrix": "substrate.density_matrix",
    "qubit_param": "substrate.qubit_param", "qubit_ema": "substrate.qubit_ema",
    "cluster": "substrate.cluster", "cumulant_cluster": "substrate.cumulant_cluster",
    "product_cluster": "substrate.product_cluster", "hamiltonian": "substrate.hamiltonian",
    "field": "substrate.field", "field_gauge": "substrate.field_gauge",
    "field_growth": "substrate.field_growth", "field_unify": "substrate.field_unify",
    "belavkin": "substrate.belavkin", "collapse": "substrate.collapse",
    "continuous_geometry": "substrate.continuous_geometry",
    "batched_evolve": "substrate.batched_evolve", "substrate": "substrate.backend",
    "classical_reservoir": "substrate.classical_reservoir", "fractal": "substrate.fractal",
    "fractal_stack": "substrate.fractal_stack", "similarity": "substrate.similarity",
    "population": "substrate.population", "params": "substrate.params",
    "param_bundles": "substrate.param_bundles", "web_topology": "substrate.web_topology",
    "world_graph": "substrate.graph", "world_model": "substrate.ground",
    # projection
    "gauge": "projection.gauge", "gauge_name": "projection.gauge_name",
    "emoji": "projection.emoji", "shelf": "projection.shelf",
    "graph_state": "projection.graph_state", "transparency": "projection.transparency",
    # clocks / learning / foresight (incl. forward targets for lazy imports)
    "phi_clock": "clocks.phi_clock", "adaptive_clock": "clocks.adaptive_clock",
    "cadence_dial": "clocks.cadence_dial", "compute_scheduler": "clocks.compute_scheduler",
    "berry_tape": "clocks.berry_tape",
    "meta_idioms": "learning.meta_idioms", "universal_learner": "learning.universal_learner",
    "stream_tape": "learning.stream_tape", "surprise_tape": "learning.surprise_tape",
    "competence": "learning.competence", "autonomy": "learning.autonomy",
    "agency": "learning.agency", "attention": "learning.attention",
    "calibration": "learning.calibration", "training": "learning.training",
    "training_backbone": "learning.training_backbone", "regressor": "learning.regressor",
    "observation_trust": "learning.observation_trust", "context": "learning.context",
    "context_learning": "learning.context_learning", "coupling_learn": "learning.coupling_learn",
    "learning_router": "learning.learning_router", "confounding": "learning.confounding",
    "teacher": "learning.teacher", "approach": "learning.approach",
    "seed_profile": "learning.seed_profile", "embedder": "learning.embedder",
    "reward.registry": "learning.reward.registry", "reward.channel": "learning.reward.channel",
    "trust_web": "foresight.trust_web", "qubit_trust_web": "foresight.qubit_trust_web",
    "predictions": "foresight.predictions", "leaf_forecast": "foresight.leaf_forecast",
    "forecast_scopes": "foresight.forecast_scopes", "forecast_surface": "foresight.forecast_surface",
    "forecast_rollout": "foresight.forecast_rollout",
    "forecast_comprehension": "foresight.forecast_comprehension",
    "bpu_forecast": "foresight.bpu_forecast", "bpu_dispatch": "foresight.bpu_dispatch",
    "dreaming": "foresight.dreaming", "dream_loop": "foresight.dream_loop",
    "dream_topology": "foresight.dream_topology",
    # membranes / root
    "tendril": "membranes.tendril", "output_surface": "membranes.egress",
    "reservoir": "engine", "bootstrap": "boot", "runner": "learning.runner",
    "house_spec": "spec.schema",
    # special: gauge.py's one import from solar_clock is pure math that moves to bloch
    "solar_clock": "substrate.bloch",
}

# meerkat.<pkg> absolute import prefixes -> umwelt module
CORE_MAP = {
    "meerkat.core.util": "umwelt._util",
    "meerkat.core.models": "umwelt.events",
    "meerkat.core.event_replay": "umwelt.events",
    "meerkat.sensors.bridge": "umwelt.spec.roles",   # role classifiers; normalizers hand-split
}

# Phase 1 manifest: (source path relative to MEERKAT, dest module key in MAP)
P1 = [
    "bloch", "density_matrix", "qubit_param", "qubit_ema", "cluster", "cumulant_cluster",
    "product_cluster", "hamiltonian", "field", "field_gauge", "field_growth", "field_unify",
    "belavkin", "collapse", "continuous_geometry", "batched_evolve", "substrate",
    "classical_reservoir", "fractal", "fractal_stack", "similarity", "population", "params",
    "param_bundles", "web_topology", "world_graph", "world_model",
    "gauge", "gauge_name", "emoji", "shelf", "phi_clock", "meta_idioms",
]

# Files already hand-curated — never overwritten by re-runs (grow this list as curation lands).
# The ENTIRE P1 set is frozen: every file was at least vocabulary-reworded, several deeply
# curated (field, fractal_stack, param_bundles, graph, ground, field_unify, field_growth,
# gauge, gauge_name, emoji, shelf). Re-extract with --force only if re-syncing from meerkat
# deliberately, then re-apply the curation (see tools/RENAMES.md).
FROZEN: set[str] = {
    "substrate/bloch.py", "substrate/density_matrix.py", "substrate/qubit_param.py",
    "substrate/qubit_ema.py", "substrate/cluster.py", "substrate/cumulant_cluster.py",
    "substrate/product_cluster.py", "substrate/hamiltonian.py", "substrate/field.py",
    "substrate/field_gauge.py", "substrate/field_growth.py", "substrate/field_unify.py",
    "substrate/belavkin.py", "substrate/collapse.py", "substrate/continuous_geometry.py",
    "substrate/batched_evolve.py", "substrate/backend.py", "substrate/classical_reservoir.py",
    "substrate/fractal.py", "substrate/fractal_stack.py", "substrate/similarity.py",
    "substrate/population.py", "substrate/params.py", "substrate/param_bundles.py",
    "substrate/web_topology.py", "substrate/graph.py", "substrate/ground.py",
    "projection/gauge.py", "projection/gauge_name.py", "projection/emoji.py",
    "projection/shelf.py", "clocks/phi_clock.py", "learning/meta_idioms.py",
}

_FROM_REL = re.compile(r"^(\s*)from \.([a-zA-Z_.]+) import (.+)$")
_FROM_PKG = re.compile(r"^(\s*)from \. import (.+)$")
_FROM_ABS = re.compile(r"^(\s*)from meerkat\.brain\.([a-zA-Z_.]+) import (.+)$")
_FROM_CORE = re.compile(r"^(\s*)from (meerkat\.[a-zA-Z_.]+) import (.+)$")


def dest_path(mod: str) -> Path:
    return SRC / (MAP[mod].replace(".", "/") + ".py")


def rewrite_line(line: str, unresolved: list[str], lineno: int) -> str:
    m = _FROM_REL.match(line)
    if m:
        indent, name, rest = m.groups()
        if name in MAP:
            return f"{indent}from umwelt.{MAP[name]} import {rest}"
        unresolved.append(f"  L{lineno}: {line.strip()}")
        return line
    m = _FROM_PKG.match(line)
    if m:
        indent, names = m.groups()
        out = []
        for part in names.split(","):
            bits = part.strip().split(" as ")
            name = bits[0].strip()
            alias = bits[1].strip() if len(bits) > 1 else None
            if name not in MAP:
                unresolved.append(f"  L{lineno}: from . import {part.strip()}")
                out.append(f"{indent}from . import {part.strip()}")
                continue
            pkg, _, leaf = MAP[name].rpartition(".")
            want = alias or name
            as_clause = f" as {want}" if leaf != want else ""
            out.append(f"{indent}from umwelt.{pkg} import {leaf}{as_clause}")
        return "\n".join(out)
    m = _FROM_ABS.match(line)
    if m:
        indent, name, rest = m.groups()
        if name in MAP:
            return f"{indent}from umwelt.{MAP[name]} import {rest}"
        unresolved.append(f"  L{lineno}: {line.strip()}")
        return line
    m = _FROM_CORE.match(line)
    if m:
        indent, name, rest = m.groups()
        if name in CORE_MAP:
            return f"{indent}from {CORE_MAP[name]} import {rest}"
        unresolved.append(f"  L{lineno}: {line.strip()}")
        return line
    return line


def extract(mod: str, force: bool = False) -> list[str]:
    src = MEERKAT / BRAIN / (mod + ".py")
    dst = dest_path(mod)
    rel = str(dst.relative_to(SRC))
    if rel in FROZEN and not force:
        return [f"{mod}: FROZEN (hand-curated), skipped"]
    text = src.read_text()
    unresolved: list[str] = []
    lines = [rewrite_line(l, unresolved, i + 1) for i, l in enumerate(text.splitlines())]
    out = "\n".join(lines) + "\n"
    out = out.replace("MEERKAT_", "UMWELT_")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(out)
    report = [f"{mod} -> src/umwelt/{rel}"]
    if unresolved:
        report.append(f"  UNRESOLVED ({len(unresolved)}):")
        report.extend(unresolved)
    return report


def main() -> None:
    force = "--force" in sys.argv
    todo = P1
    print(f"extracting {len(todo)} modules from {MEERKAT}")
    hand_work = 0
    for mod in todo:
        for line in extract(mod, force=force):
            if "UNRESOLVED" in line:
                hand_work += 1
            print(line)
    print(f"\n{len(todo)} modules copied; {hand_work} need hand-curation (see UNRESOLVED above)")


if __name__ == "__main__":
    main()
