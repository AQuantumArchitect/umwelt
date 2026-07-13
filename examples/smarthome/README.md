# Smart home — the origin domain

**Status: deployed.** This is where the engine comes from: **meerkat**, a home-
comprehension system running for 18 months on a $100 ARM board in a real apartment with
a real resident — real Zigbee/Wi-Fi sensor lanes, real light/climate actuators, a
~1,450-test release gate, autonomy earned per-output. Meerkat remains a separate
deployment (this example never imports it); what lives HERE is the domain's vocabulary,
registered the way any umwelt domain registers itself:

- [`solar.py`](solar.py) — the `SolarDriver`: local apparent solar time (longitude +
  equation-of-time), the canonical domain-registered `PeriodicDriver`.
- [`vocabulary.py`](vocabulary.py) — every registration the home makes: role input
  modes (motion is unitary, temperature is dissipative-analog), home normalizers
  (`temp_f`, `contact_presence`), render glyphs, the sleep outcome channel, the
  known-place token, the solar clock.

```python
from examples.smarthome.vocabulary import register_smarthome_vocabulary
register_smarthome_vocabulary()
# then author a home DomainSpec exactly as docs/SPEC.md describes
```

This directory is the vocabulary-registration reference, not a starting template —
there is no `spec.py`/proof/tests here to copy. If you're starting a brand-new domain,
begin at [`examples/gridworld/`](../gridworld) (complete and standalone) and
[docs/NEW_DOMAIN.md](../../docs/NEW_DOMAIN.md) (the checklist); come back here
specifically for how `role_modes`/normalizers/a custom driver get registered.

The measured evidence cited in [CLAIMS.md](../../CLAIMS.md) (the 24-day de-confounding
A/B, the ladder-walk verdict, the 18-month deployment) was produced on meerkat's real
tapes and lives with them.
