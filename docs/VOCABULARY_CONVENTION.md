# Vocabulary convention — registering a domain's emoji

`umwelt.projection.emoji`'s contract: "the engine ships only the neutral fallback ... registered
by the domain." Nobody enforced that contract, so it drifted — five separate worlds shipped with
roles or nodes silently falling through to the neutral 🔵/⚪/🔴/📦 default (`hive-ops`,
`project_membrane`, `yurt-mood`, `inbox_world`'s node icons, found 2026-08-02). Every real
`vocabulary.py` this project has (`examples/smarthome/`, `worlds/hive_ops/`, `worlds/roc_self/`,
`worlds/spacewheat_self/`, `worlds/yurt_mood/`, ...) already agrees on the shape below; this just
writes it down so the next one doesn't have to reverse-engineer it from five other files, and so
`tests/test_world_vocabulary_coverage.py` can point here in its failure messages.

## The shape

```python
register_role_emoji("truthful", {"pos": "✅", "zero": "❔", "neg": "🤥", "coherent": "⚡"})
```

- **`pos` / `neg`** — the axis's two fixed poles. `pos` is the **warm pole**, and is listed /
  thought-of first (the "strand convention" most worlds' docstrings reference directly). Warm =
  healthy/present/flowing, not necessarily "high number" — check the world's own WIRE SIGN LAW
  comment for how warmth maps to Bloch-z before picking which pole is which.
- **`zero`** — the mid/uncertain state. `❔` is the common default across the yurt worlds (mostly
  ceremony/coordination axes with no natural third state); `examples/smarthome/vocabulary.py` (the
  canonical worked example) instead uses a genuinely meaningful zero glyph per role (🚪 for
  presence, 🧘 for activity) where one exists. Prefer a meaningful zero when the domain has one;
  `❔` is the honest fallback, not a rule to follow blindly.
- **`coherent`** — the transverse-coherence modifier. `⚡` is what nearly every registration uses;
  name it as the de facto standard. Not absolute — `examples/smarthome/vocabulary.py`'s own
  `temperature` role uses `🌀` where it reads better for that specific axis. Deviate when it's a
  real fit, not by default.

## Transcribe, don't invent

When a glyph pair is designed somewhere else — a docstring that already spells it out, a sibling
system's own emoji table — **cite the source in a code comment** and use those glyphs verbatim.
`worlds/yurt_mood/vocabulary.py` is the example: its own `world.py` docstring had already designed
`activity`/`confidence`/`coordination`'s glyphs in prose, and `blocked`/`freshness` were pulled
directly from `Claura/quantum_lichen/emoji_language.py`'s `stability`/`novelty` axis table — nothing
guessed.

When there is genuinely **nothing to transcribe from** — an unfamiliar domain with no docstring, no
sibling table, no prior art (BAR's `verdance_campaign`/`verdance_proxy` worlds, as of 2026-08-02) —
that is an honest gap, not a license to invent plausible-looking glyphs. `project_membrane`'s own
docstring already names this doctrine: *"an honest fog beats a fake warmth."* Add the world to
`NO_VOCAB_ALLOWLIST` in `tests/test_world_vocabulary_coverage.py` with a comment naming what's
missing, and let a real glyph land once someone who knows the domain can transcribe it.

## Node icons render everywhere (since 2026-08-02)

`register_node_icon(name, icon)` feeds `emoji.field_summary()` (text/console) AND the 3D field:
`cognifold_trace` embeds each register's `node_icon` in the trace itself, and SpaceWheat's
`CognifoldForecastField.gd` draws it as a badge beside every belief — the renderer never touches
this registry; the icon travels with the data. One registration covers both views. The full
channel legend for the 3D field lives at `SpaceWheat/docs/COGNIFOLD_CHANNELS.md`.

## Dynamically-discovered node sets

A world whose node set is discovered at runtime rather than fixed (`worlds/inbox_world/world.py`'s
`build_spec(senders)` — "a real inbox's sender set is unbounded") can only register icons for the
names it knows about *now* (its fixture/seed set). That's not a gap to chase forever — a genuinely
new node name falling back to the neutral icon is honest, the same way an unrecognized role would
be. Register what's known; don't block on making the unknowable knowable.

## Enforcement

`umwelt/tests/test_world_vocabulary_coverage.py` walks every world under the yurt tree (or wherever
`YURT_WORLDS_DIR` points), imports each in isolation, and asserts every role every `NodeSpec`
declares is in `ROLE_EMOJI` and every node name is in `NODE_ICONS` — except the explicit
`NO_VOCAB_ALLOWLIST` entries above. Skips cleanly (not a failure) when the yurt tree isn't mounted
in the environment running the suite.
