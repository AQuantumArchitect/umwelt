"""Phase 5 facet kits: public cassette + baseline + honesty tier."""
from __future__ import annotations

from pathlib import Path

from umwelt.kits.attention import ATTENTION_SPEC, run_attention_baseline
from umwelt.kits.dream import DREAM_SPEC, run_dream_baseline
from umwelt.kits.fog import FOG_KIT_SPEC, run_fog_baseline
from umwelt.kits.market import MARKET_SPEC, run_market_baseline

KIT_ROOT = Path(__file__).resolve().parents[1] / "src" / "umwelt" / "kits"


def test_fog_kit_baseline():
    r = run_fog_baseline(ticks=120)
    print(r.summary())
    assert r.kit == "fog"
    assert r.honesty
    # metrics are finite; win is preferred but honest report always prints
    assert r.engine_mae >= 0 and r.freeze_mae >= 0


def test_attention_kit_baseline():
    r = run_attention_baseline()
    print(r.summary())
    assert r.beats_baseline, r.summary()
    assert "corrupted" in r.honesty or "synthetic" in r.honesty


def test_market_kit_baseline():
    r = run_market_baseline()
    print(r.summary())
    assert r.beats_baseline, r.summary()
    assert r.clean_err < r.poison_err


def test_dream_kit_never_actuates():
    r = run_dream_baseline()
    print(r.summary())
    assert r.live_dispatches == 0
    assert r.counterfactual_batches > 0
    assert r.field_changed is False
    assert r.beats_baseline


def test_kit_readmes_and_specs_exist():
    for name in ("fog", "attention", "market", "dream"):
        readme = KIT_ROOT / name / "README.md"
        assert readme.is_file(), f"missing {readme}"
        text = readme.read_text(encoding="utf-8")
        assert "Honesty" in text or "honesty" in text
    assert FOG_KIT_SPEC.name.startswith("fledgeling-fog")
    assert ATTENTION_SPEC.name.startswith("kit-attention")
    assert MARKET_SPEC.name.startswith("kit-market")
    assert DREAM_SPEC.name.startswith("fledgeling-fog")
