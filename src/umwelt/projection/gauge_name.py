"""Gauge-naming — a brain artifact's name is the PARAMETRIC output of its gauge.

A brain is a point in (parameter-space) × (datastream-metadata). Instead of a hand-typed
`magic_aura_farming.v3.09`, the name is COMPUTED deterministically from that point, so every distinct gauge
gets its own readable, sortable name that carries the same information the lineage DAG does:

    {recipe}.{place}.{surface}.{corpus}.{gauge}.g{gen}-{fp}
    e.g. magic.9v6kz.dev6-3c1d.seed.L0A1d0.g0-7f2a

  recipe   — the mint recipe / seed origin (magic | blank | …); the parameter-space ORIGIN.
  place    — where the engine thinks it is (anchor gear decoded → known place or geohash5; nowhere=unlocated).
  surface  — the body/sensor-actuator set signature (dev{N}-{hash4} over the sorted binding ids).
  corpus   — the datastream it was trained on (cassette basename or ds-{hash4}; seed = recipe-minted).
  gauge    — the ContextState axes: L{learn}A{actuate}d{dt_rung}. "Each gauge has its own name."
  gen      — lineage generation (depth from fresh in the shelf DAG).
  fp       — 4-char fingerprint of the rounded learned param fiber (the content-addressed disambiguator).

The name is computed at MINT time and stored as the shelf record's `label` (beside `topo_signature` for
boot/select compatibility). Pure + deterministic: same (params × datastream) → same name. See
experiments/mint_magic.py + the boot/console brain-picker.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math

logger = logging.getLogger(__name__)

_PHI = (1.0 + 5.0 ** 0.5) / 2.0
_GEOHASH32 = "0123456789bcdefghjkmnpqrstuvwxyz"   # standard geohash base-32 alphabet


def _hash4(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:4]


def _slug(s: str) -> str:
    """Lowercase, keep [a-z0-9_-], drop the rest — so a free-text component can't break the dotted
    format (the structured tokens surface/gauge/fp are already controlled and pass through compose raw)."""
    out = [c if (c.isalnum() or c in "_-") else "" for c in str(s).lower()]
    return "".join(out) or "x"


def geohash5(lat: float, lon: float, precision: int = 5) -> str:
    """Standard geohash encode to `precision` chars (5 ≈ ±2.4km cell) — a deterministic place token."""
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    bits, bit, ch, out, even = 0, 0, 0, [], True
    while len(out) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon >= mid:
                ch |= (1 << (4 - bit)); lon_lo = mid
            else:
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat >= mid:
                ch |= (1 << (4 - bit)); lat_lo = mid
            else:
                lat_hi = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            out.append(_GEOHASH32[ch]); bit = 0; ch = 0
    return "".join(out)


# Known places (anchors) → a friendly token instead of a geohash. EMPTY by default:
# the engine ships no geography; a domain registers its anchors (register_known_place).
_KNOWN_PLACES: list[tuple[str, float, float, float]] = []   # (name, lat, lon, deg-radius)


def register_known_place(name: str, lat: float, lon: float, deg_radius: float = 0.25) -> None:
    """Register a friendly place token minted when the decoded anchor lands within
    `deg_radius` degrees of (lat, lon). Without a registration the token is a geohash."""
    _KNOWN_PLACES.append((_slug(name), float(lat), float(lon), float(deg_radius)))


def _place(reservoir) -> str:
    """The engine's decoded location as a name: a known place if near one, else geohash5,
    else nowhere (unlocated). Place is PROVENANCE, not radius: only an anchor grounded
    by evidence (ground_anchor, or restored from an artifact that carries a fix) may
    mint a token — long runs of field dynamics can drift an un-grounded qubit off
    maximally-mixed, and that drift is not a coordinate anyone gave the engine
    (found by a 13-day real-deployment replay; the 1-day proof never drifts that far)."""
    try:
        if "geo" not in getattr(reservoir, "_grounded_anchors", ()):
            return "nowhere"
        b = reservoir.location_bloch()
        if math.sqrt(sum(c * c for c in b)) < 0.5:      # maximally-mixed earth gear → blank floor
            return "nowhere"
        lat, lon = reservoir.location_latlon()
        for name, plat, plon, rad in _KNOWN_PLACES:
            if abs(lat - plat) <= rad and abs(lon - plon) <= rad:
                return name
        return geohash5(lat, lon)
    except Exception:
        return "nowhere"


def _surface(reservoir) -> str:
    """Body signature: device count + a hash over the sorted actuator+sensor binding ids."""
    try:
        act = sorted(getattr(reservoir.actuator_bridge, "bindings", {}) or {})
        sen = sorted(getattr(reservoir.sensor_bridge, "bindings", {}) or {})
        return f"dev{len(act)}-{_hash4('|'.join(act) + '#' + '|'.join(sen))}"
    except Exception:
        return "dev0-0000"


def _gauge(reservoir) -> str:
    """The ContextState axes as a token: L{learn}A{actuate}d{dt_rung}."""
    try:
        learn = int(round(reservoir._root_param("context_learn", 1.0)))
        actuate = int(round(reservoir._root_param("context_actuate", 1.0)))
        dt = float(reservoir._root_param("context_dt_factor", 1.0))
        rung = int(math.floor(math.log(max(dt, 1.0), _PHI)))
        return f"L{learn}A{actuate}d{rung}"
    except (AttributeError, TypeError, ValueError) as e:
        # b9.64: a mis-gauged brain must not SILENTLY mint a plausible coordinate — the
        # fallback token stays (a name must always mint) but the failure is now visible.
        logger.warning("gauge token unreadable (%s) — minting Lxaxdx", type(e).__name__)
        return "Lxaxdx"


def _fp(reservoir) -> str:
    """4-char fingerprint of the rounded learned param fiber (disambiguates same-coordinate brains)."""
    try:
        snap = reservoir._snapshot_param_fiber()
        rounded = {
            node: {k: (round(float(v[0]), 3) if isinstance(v, (list, tuple)) else round(float(v), 3))
                   for k, v in (params or {}).items()}
            for node, params in (snap or {}).items()
        }
        return _hash4(json.dumps(rounded, sort_keys=True))
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning("param-fiber fingerprint unreadable (%s) — minting 0000", type(e).__name__)
        return "0000"


def _recipe(reservoir) -> str:
    return _slug(getattr(reservoir, "mint_recipe", None) or getattr(reservoir, "seed_profile", "blank"))


def _corpus(lineage: dict | None) -> str:
    """The datastream the brain trained on: a cassette basename, or ds-{hash4} of cassette parents, or
    'seed' (recipe-minted, no cassette)."""
    if not lineage:
        return "seed"
    cas = lineage.get("cassette") or lineage.get("cassettes")
    if isinstance(cas, (list, tuple)):
        cas = cas[0] if cas else None
    if cas:
        base = str(cas).rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return _slug(base)
    pcas = lineage.get("cassette_parents") or []
    if pcas:
        return "ds-" + _hash4("|".join(str(p) for p in pcas))
    return "seed"


def compose_name(*, recipe, place, surface, corpus, gauge, gen, fp) -> str:
    """The pure formatter — the dotted gauge-name from its components. Free-text fields (recipe/place/
    corpus) are slugged; the structured tokens (surface/gauge/fp) are generated controlled and pass raw."""
    return f"{_slug(recipe)}.{_slug(place)}.{surface}.{_slug(corpus)}.{gauge}.g{int(gen)}-{fp}"


def gauge_name(reservoir, *, lineage: dict | None = None) -> str:
    """The parametric name for a brain reservoir = (parameter-space signature) × (datastream metadata).
    `lineage` (optional) carries datastream provenance: {cassette|cassettes, cassette_parents, gen}."""
    gen = int((lineage or {}).get("gen", 0))
    return compose_name(
        recipe=_recipe(reservoir), place=_place(reservoir), surface=_surface(reservoir),
        corpus=_corpus(lineage), gauge=_gauge(reservoir), gen=gen, fp=_fp(reservoir),
    )


def topo_signature(reservoir) -> dict:
    """The topology fingerprint a loaded pickle must match the boot build on (the LAUNCH-RULE geometry
    guard). feature_dim + the cluster-kind histogram; the console warns on mismatch before a select."""
    kinds: dict[str, int] = {}
    try:
        from umwelt.substrate.backend import cluster_kind
        for c in reservoir.field.clusters.values():
            k = cluster_kind(c)
            kinds[k] = kinds.get(k, 0) + 1
    except Exception:
        pass
    return {"feature_dim": getattr(reservoir, "feature_dim", None), "cluster_kinds": kinds}
