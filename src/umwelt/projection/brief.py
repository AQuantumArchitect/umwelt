"""brief — the universal human read of ANY world, composed from wire dicts.

Pure functions over the worker's own HTTP payloads (`/health`, `/cognifold`,
`/recommendations`). Deliberately NO engine import and NO vocabulary import:
everything a brief needs already travels IN the trace (north/south emoji,
node_icon, reliability, surprise) — the same law that put those fields there.
A client that only ever sees the JSON renders the same brief the server would;
a world this process has never heard of still reads honestly.

The human text and the machine dict come from the same compose step
(`compose_brief` → `render_brief`), never two divergent sources.
"""
from __future__ import annotations

from datetime import datetime, timezone

from umwelt.projection.emoji import field_summary

_Z_THRESHOLD = 0.3          # same pole threshold as qubit_emoji
_COHERENT_GLYPH = "💠"      # engine-neutral modifier (trace carries no domain coherent glyph)
_LOW_TRUST = 0.3            # reliability (observation-trust α) below this reads as distrusted


def _state_glyph(reg: dict) -> str:
    """Where the belief SITS on its axis, from trace fields only: the pole emoji the
    trace itself carries, the neutral ⚪ between poles, + a coherence modifier."""
    z = 2.0 * float(reg.get("value", reg.get("p0", 0.5))) - 1.0
    if z > _Z_THRESHOLD:
        glyph = str(reg.get("north_emoji") or "🔵")
    elif z < -_Z_THRESHOLD:
        glyph = str(reg.get("south_emoji") or "🔴")
    else:
        glyph = "⚪"
    if float(reg.get("r_xy", 0.0)) > 0.5:
        glyph += _COHERENT_GLYPH
    return glyph


def _age_s(iso_ts: str | None) -> float | None:
    if not iso_ts:
        return None
    try:
        ts = datetime.fromisoformat(str(iso_ts))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())
    except ValueError:
        return None


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "never/unknown"
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 129600:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def compose_brief(health: dict, trace: dict, recommendations: list | None = None) -> dict:
    """One dict holding everything the brief shows — the machine face IS this."""
    regs = [r for r in trace.get("registers", []) if isinstance(r, dict)]

    nodes: dict[str, dict] = {}
    for r in regs:
        n = nodes.setdefault(str(r.get("node", "?")), {
            "icon": str(r.get("node_icon", "📍")), "roles": [], "purities": []})
        n["roles"].append({
            "role": str(r.get("role", "?")),
            "glyph": _state_glyph(r),
            "value": round(float(r.get("value", 0.5)), 3),
            "confidence": round(float(r.get("confidence", 0.0)), 3),
            "reliability": r.get("reliability"),
            "surprise": r.get("surprise"),
        })
        n["purities"].append(float(r.get("purity", 1.0)))

    # movers: the beliefs most worth a human's eyes right now — surprised first,
    # then the most committed poles. Honest empty when the field is quiet.
    surprised = sorted(
        (r for r in regs if r.get("surprise") is not None),
        key=lambda r: float(r["surprise"]), reverse=True)
    movers = [{"belief": f"{r.get('node')}.{r.get('role')}",
               "surprise": round(float(r["surprise"]), 4)} for r in surprised[:3]
              if float(r["surprise"]) > 1e-6]
    if not movers:
        poles = sorted(regs, key=lambda r: abs(2 * float(r.get("value", 0.5)) - 1),
                       reverse=True)
        movers = [{"belief": f"{r.get('node')}.{r.get('role')}",
                   "value": round(float(r.get("value", 0.5)), 3)}
                  for r in poles[:3]
                  if abs(2 * float(r.get("value", 0.5)) - 1) > 0.8]

    low_trust = [f"{r.get('node')}.{r.get('role')}" for r in regs
                 if r.get("reliability") is not None
                 and float(r["reliability"]) < _LOW_TRUST]

    mi_edges = [e for e in trace.get("edges", [])
                if isinstance(e, dict) and e.get("kind") == "mi"]

    return {
        "world": str(trace.get("world") or health.get("world") or "?"),
        "step": health.get("step"),
        "last_event_age_s": _age_s(health.get("last_event_ts")),
        "last_event_ts": health.get("last_event_ts"),
        "registers": len(regs),
        "nodes": nodes,
        "movers": movers,
        "low_observation_trust": low_trust,
        "live_mi_edges": len(mi_edges),
        "shadow_recommendations": list(recommendations or []),
    }


def render_brief(brief: dict) -> str:
    """The human face — formatted FROM compose_brief's dict, nothing else."""
    lines = [
        f"world {brief['world']} — step {brief.get('step')} — "
        f"{brief['registers']} beliefs — last event {_fmt_age(brief.get('last_event_age_s'))}"
    ]
    node_emojis = {n: " ".join(r["glyph"] for r in d["roles"])
                   for n, d in brief["nodes"].items()}
    node_purities = {n: (sum(d["purities"]) / len(d["purities"]) if d["purities"] else 1.0)
                     for n, d in brief["nodes"].items()}
    node_icons = {n: d["icon"] for n, d in brief["nodes"].items()}
    lines.append(field_summary(node_emojis, node_purities, node_icons=node_icons))

    if brief["movers"]:
        parts = [f"{m['belief']}"
                 + (f" (surprise {m['surprise']})" if "surprise" in m else f" (value {m['value']})")
                 for m in brief["movers"]]
        lines.append("  movers: " + ", ".join(parts))
    if brief["low_observation_trust"]:
        lines.append("  low observation-trust: " + ", ".join(brief["low_observation_trust"]))
    if brief["live_mi_edges"]:
        lines.append(f"  live correlation: {brief['live_mi_edges']} mi edge(s) firing")
    recs = brief["shadow_recommendations"]
    if recs:
        lines.append(f"  shadow (would act, didn't dispatch): {len(recs)}")
        for r in recs[:5]:
            if isinstance(r, dict):
                # the worker's wire shape: actuator_id/command/node/role/reason
                name = r.get("actuator_id") or r.get("name") or "?"
                act = r.get("command", r.get("recommendation", r.get("value")))
                driver = f"{r.get('node')}.{r.get('role')}" if r.get("node") else ""
                lines.append(f"    · {name} → {act}" + (f"  (driven by {driver})" if driver else ""))
    return "\n".join(lines)
