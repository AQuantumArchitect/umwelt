# Kit: fog

**Honesty tier:** synthetic CI cassette — scout partial observation.  
**Roadmap:** [docs/FLEDGELING_CORE.md](../../../docs/FLEDGELING_CORE.md) Phase 5.

Partial observation / scout η over a corridor graph. Reuses the Phase-1
`examples/fledgeling_fog` domain via the host API.

```python
from umwelt.kits.fog import run_fog_baseline
print(run_fog_baseline().summary())
```

**Not proven:** multi-agent fog as a product feature, narrative discovery, live
game integration (Phase 6). Multi-mind privacy is gated separately under
`tests/test_multimind_privacy.py`.
