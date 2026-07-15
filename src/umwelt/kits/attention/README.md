# Kit: attention (Warmth-lite)

**Honesty tier:** synthetic CI — two sources, one corrupted.

Two channels into one `signal` role. The corrupted source has low η; a naive
full-confidence merge is the baseline to beat via isolation / low-η weighting.

```python
from umwelt.kits.attention import run_attention_baseline
print(run_attention_baseline().summary())
```

**Not proven:** live multi-player attention economy, catalyst dynamics, Warmth narrative.
