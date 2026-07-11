"""Competence — how much the brain has LEARNED, the signal that EARNS agency (b9.15).

The agency qubit (``agency.py``) folds from RECOMMEND toward ACT only when it is fed a
confidence ∈ [0,1] AND the operator opted in (``agency_auto_act_enabled``). Before b9.15 that
confidence was a PLACEHOLDER — an occupancy ``alpha`` — so the brain climbed toward
acting on a wall-clock + a switch, NOT because it had proven it learned the world. This module
closes that gap: competence is the brain's own learnedness, read from the SAME reload-surviving
quantities the transparency lens (``transparency.py``, b9.10) already exposes, multiplied by its
live prediction skill::

    competence = learnedness × skill            (the conservative AND)

  • learnedness ∈ [0,1] — the mean ``settled`` ( = 1 − σ/σ_prior ) across the learnable param
    fiber: how far each posterior width has shrunk from its prior (1 = converged, 0 = untouched).
    σ is pickled, so this SURVIVES a forebrain reload — a re-deployed learned brain reads
    competent immediately, while a fresh blank-slate brain reads ~0 (every param still sits at its
    prior width). This is the honest "has it learned" axis (drift survives reload; update_count
    does not — the b9.10.1 lesson).

  • skill ∈ [0,1] — ``1 − clipped(surprise_ema)``: how UNSURPRISED the brain is on the live world.
    A model that converged on the WRONG thing keeps high surprise → low skill → no agency.
    Accuracy by fluke (low surprise, nothing settled) → low learnedness → no agency. Only a brain
    that has BOTH settled AND stopped being surprised earns the fold toward ACT.

Read-only and cheap (one O(params) graph walk, all O(1) reads — no density-matrix rebuild), so it
is safe to call every tick. The agency tick consumes ``competence_score(reservoir)`` as its
confidence; ``transparency.model_snapshot`` surfaces the breakdown so the operator can WATCH agency
be earned. See [[project_b910_transparency]], agency.py, qubit_param.py.
"""
from __future__ import annotations

from typing import Any

from umwelt._util import clamp01 as _clip01  # the shared unit clamp (M6)


def _graph(reservoir: Any):
    field = getattr(reservoir, "field", None)
    return getattr(field, "graph", None) or getattr(reservoir, "graph", None)


def _param_settled(p: Any) -> float | None:
    """A learnable param's settledness ∈ [0,1] = 1 − σ/σ_prior (same definition the transparency
    lens uses). None when the param has no prior width (not a learnable fiber qubit) or is frozen —
    those are excluded from the learnedness mean so a pile of structural constants can't dilute it."""
    prior_sigma = getattr(p, "prior_sigma", None)
    if not prior_sigma:
        return None
    if getattr(p, "frozen", False):
        return None
    try:
        return _clip01(1.0 - (float(p.sigma) / float(prior_sigma)))
    except (TypeError, ValueError, AttributeError, ZeroDivisionError):
        # b9.64: narrowed — a malformed σ reads as not-learnable (excluded), never as settled
        return None


def learnedness(reservoir: Any) -> float:
    """Mean ``settled`` across the learnable param fiber ∈ [0,1]. 0 = blank slate (every posterior
    still at its prior width); → 1 as the Kalman updates shrink the widths. Reload-surviving (σ is
    pickled). De-dupes shared archetype bundles by identity (per-node device bundles are one
    object) so a populous topology doesn't over-weight a repeated archetype."""
    graph = _graph(reservoir)
    if graph is None:
        return 0.0
    try:
        nodes = list(graph.all_nodes())
    except (AttributeError, TypeError):
        # b9.64: narrowed — only a graph that can't walk reads as blank; anything else raises
        return 0.0
    seen: set[int] = set()
    vals: list[float] = []
    for node in nodes:
        b = getattr(node, "param_bundle", None)
        if b is None or id(b) in seen:
            continue
        seen.add(id(b))
        for _, p in (getattr(b, "params", {}) or {}).items():
            s = _param_settled(p)
            if s is not None:
                vals.append(s)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _surprise_ema(reservoir: Any) -> float | None:
    """The brain's live aggregate prediction surprise (substrate EMA) — the SAME signal the adaptive
    clock + stream-learn logger read (``fractal_stack.scales[0]._surprise_ema``). None when no
    learner has run yet."""
    fs = getattr(reservoir, "fractal_stack", None)
    if fs is not None and getattr(fs, "scales", None):
        try:
            v = getattr(fs.scales[0], "_surprise_ema", None)
            if v is not None:
                return float(v)
        except (TypeError, ValueError, IndexError):   # b9.64: narrowed from except-Exception
            return None
    return None


def prediction_skill(reservoir: Any) -> float:
    """How UNSURPRISED the brain is on the live world ∈ [0,1] = 1 − clip(surprise_ema). Neutral 0.5
    when no learner has reported yet (accuracy is unproven — the brain hasn't earned skill, but we
    don't punish a cold start either; learnedness still floors a fresh brain near 0)."""
    surp = _surprise_ema(reservoir)
    if surp is None:
        return 0.5
    return _clip01(1.0 - _clip01(abs(surp)))


def competence_score(reservoir: Any) -> float:
    """The single confidence ∈ [0,1] the agency tick consumes: learnedness × skill (the conservative
    AND). Fresh brain → ~0 (won't take over); a brain that has both SETTLED and stopped being
    SURPRISED → high (earns the fold toward ACT)."""
    return _clip01(learnedness(reservoir) * prediction_skill(reservoir))


def competence_snapshot(reservoir: Any) -> dict:
    """The legible breakdown for the transparency page — so the takeover is observable, not magic."""
    learn = learnedness(reservoir)
    skill = prediction_skill(reservoir)
    snap = {
        "competence": round(_clip01(learn * skill), 4),
        "learnedness": round(learn, 4),
        "skill": round(skill, 4),
    }
    try:
        per = actuator_competence(reservoir)
    except Exception:
        per = {}
    if per:
        snap["actuators"] = per
    return snap


# ── per-actuator competence (b9.44): the SAME law, resolved to each tendril family ────────
#
# The global scalar answers "has the brain learned the world"; the Watch→Run switch is per
# ACTUATOR, so each one should EARN its own flip from its own evidence:
#
#     competence_a = learnedness(the family's OWN fiber params) × skill(the roles it confounds)
#
# Both factors are graph-derived, no per-actuator hand-coding: the param family is the union
# of the member tendrils' DEFAULTS (the b9.9 dissolution put every tendril's constants on the
# root fiber under its own prefix), and the confounded roles come from
# confounding.actuator_confounding via each tendril's graph_node — the same map the learning
# router already trusts. Posture flags (the *_enabled switches the autonomy surface flips at
# alpha=1) are EXCLUDED from learnedness: flipping a switch is a decision, not learning, and
# a hard set shrinks sigma so it would read as instantly "settled".


def _registry_members(reservoir: Any, entry: Any) -> list:
    """The live tendrils belonging to one autonomy-REGISTRY entry: membership = the entry's
    enable flag appears in the tendril class's DEFAULTS (the family's own gate param). No
    hand-coded tendril→key table; a new tendril joins its family by carrying the flag."""
    out = []
    for t in getattr(reservoir, "tendrils", None) or []:
        if entry.enable_param in (getattr(t, "DEFAULTS", None) or {}):
            out.append(t)
    return out


def _family_learnedness(reservoir: Any, members: list, exclude: set[str]) -> tuple[float | None, int]:
    """Mean settled over the family's own root-fiber params (union of member DEFAULTS keys,
    minus posture flags). None when the family has no learnable params on the bundle yet."""
    root = getattr(_graph(reservoir), "root", None)
    bundle = getattr(root, "param_bundle", None)
    params = getattr(bundle, "params", None) or {}
    names = set()
    for t in members:
        names |= set(getattr(t, "DEFAULTS", None) or {})
    names -= exclude
    vals = [s for n in sorted(names)
            if (p := params.get(n)) is not None and (s := _param_settled(p)) is not None]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def _family_skill(reservoir: Any, members: list) -> tuple[float, str]:
    """Skill at the roles this family CONFOUNDS: 1 − clip(mean per-role |production residual|
    EMA) over the members' confounded (cluster, role) targets — read from the fractal stack's
    role_surprise (the production-residual signal, not the meta-field EMA). Falls back to the
    global prediction_skill when the map or the EMA hasn't formed yet (cold start / bare
    reservoir), tagged so the readout is honest about its source."""
    fs = getattr(reservoir, "fractal_stack", None)
    graph = _graph(reservoir)
    role_surp = getattr(fs, "role_surprise", None)
    if graph is not None and callable(role_surp):
        try:
            from umwelt.learning.confounding import actuator_confounding
            conf_map = actuator_confounding(graph)
        except Exception:
            conf_map = {}
        field = getattr(reservoir, "field", None)
        clusters = getattr(field, "clusters", None) or {}
        surps: list[float] = []
        for t in members:
            hit = conf_map.get(getattr(t, "graph_node", None) or "")
            if hit is None:
                continue
            cluster_name, roles = hit
            cluster = clusters.get(cluster_name)
            vec = role_surp(cluster_name)
            if cluster is None or vec is None:
                continue
            ridx = getattr(cluster, "role_index", None) or {}
            for role in roles:
                i = ridx.get(role)
                if i is not None and i < len(vec):
                    surps.append(abs(float(vec[i])))
        if surps:
            return _clip01(1.0 - _clip01(sum(surps) / len(surps))), "roles"
    return prediction_skill(reservoir), "global"


def actuator_competence(reservoir: Any) -> dict[str, dict]:
    """{autonomy key: the per-actuator competence breakdown} for every REGISTRY entry with live
    member tendrils. Read-only and cheap (one small param walk + O(1) EMA reads); feeds the
    autonomy report / the /home Watch⇄Run switch so each actuator's earning is VISIBLE. It
    gates nothing yet — flips stay operator-owned (observe-first, the b9.3 cutover law)."""
    from umwelt.learning.autonomy import REGISTRY
    out: dict[str, dict] = {}
    for entry in REGISTRY:
        members = _registry_members(reservoir, entry)
        if not members:
            continue
        posture_flags = {entry.enable_param, *entry.extra_params}
        posture_flags |= {n for t in members
                          for n in (getattr(t, "DEFAULTS", None) or {}) if n.endswith("_enabled")}
        learn, n_params = _family_learnedness(reservoir, members, posture_flags)
        skill, skill_source = _family_skill(reservoir, members)
        comp = _clip01(learn * skill) if learn is not None else None
        out[entry.key] = {
            "competence": round(comp, 4) if comp is not None else None,
            "learnedness": round(learn, 4) if learn is not None else None,
            "skill": round(skill, 4),
            "skill_source": skill_source,
            "n_params": n_params,
            "n_tendrils": len(members),
        }
    return out
