"""umweltforge — the embedded LLM compiler + warden layer. EXPERIMENTAL.

The missing seam between "a plain-English description of a domain" and "a running
umweltd world": an embedded coding agent authors the DomainSpec module in a scoped
workspace, a deterministic gate (umwelt.spec.validate, run in a fresh subprocess)
decides whether it's real, and only a green gate registers the world. The agent's
own success claim is never trusted.

The warden is the same discipline pointed at a RUNNING world: a one-shot tick
inspects health/state/recommendations, proposes spec changes by change-type, and an
earned-autonomy dial (per-world, per-change-type, everything defaulting to
propose-only) decides whether a validated proposal may auto-apply.

The engine never imports this package; the Claude Agent SDK is an optional extra
(`pip install "umwelt-engine[forge]"`) imported lazily so the repo gate runs with
no API key and no network.
"""
