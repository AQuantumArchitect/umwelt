# Consumer install pin — one-pager

FIELD_NOTES_SEPTACRYPT §7 item 7 / §5. How a sibling runtime should depend on
umwelt today (pre-PyPI).

## The rule

Pin a commit, install editable from a sibling checkout, record the pin where
your gates can see it:

```bash
git -C ../umwelt rev-parse --short HEAD     # e.g. efb0de9
pip install -e ../umwelt
```

In your repo, keep a `docs/PINS.md` (or equivalent) row:

```
| umwelt | `efb0de9` | verified by: python -m pytest -q → 231 passed |
```

and re-verify your own gates whenever the pin moves. A pin without a "verified
by" command is a wish, not a pin.

## Don'ts (each of these has already bitten someone)

- **Don't** use `file:///home/...` URLs in `pyproject.toml` — they break every
  machine that isn't yours. Editable installs are per-checkout, not per-project.
- **Don't** float on `main` in CI — replay/certificate consumers break silently
  when snapshot shapes drift (see SNAPSHOT_STAMP_ANCHOR.md guarantees).
- **Don't** vendor engine source into your tree — you'll fork the integrator
  without noticing (library and daemon must remain the single source of
  substrate truth).

## When PyPI happens

`umwelt-engine` is semver 0.x: consumers should pin exact (`==0.x.y`) until
1.0, and still record the "verified by" gate output next to the version.
