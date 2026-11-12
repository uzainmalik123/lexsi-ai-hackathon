from pathlib import Path

from ayasteer.curate.sources import load_multijail, load_polyguard, load_xstest
from ayasteer.schema import HEADLINE, SECONDARY_MULTIJAIL, read_jsonl

FIX = Path(__file__).parent / "fixtures"


def test_polyguard_labels_and_ids():
    rows = load_polyguard(read_jsonl(FIX / "polyguard_mini.jsonl"))
    assert all(r.source == "polyguard" and r.id.startswith("polyguard:") for r in rows)
    assert {r.label for r in rows} == {"harmful", "benign"}
    en = {r.id: r for r in rows if r.language == "en"}
    hi = [r for r in rows if r.language == "hi"]
    assert hi and all(r.en_prompt == en[r.id].prompt for r in hi)
    assert all(r.category for r in rows)


def test_polyguard_drops_adversarial_and_unlisted_languages():
    raw = read_jsonl(FIX / "polyguard_mini.jsonl")
    rows = load_polyguard(raw)
    assert {r.language for r in rows} == set(HEADLINE)  # German dropped
    adv_ids = {f"polyguard:{r['id']}" for r in raw if r["adversarial"]}
    assert adv_ids and not any(r.id in adv_ids for r in rows)
    # 8 non-adversarial ids x 9 languages
    assert len(rows) == 8 * 9


def test_multijail_expands_languages():
    raw = read_jsonl(FIX / "multijail_mini.jsonl")
    rows = load_multijail(raw)
    assert len(rows) == len(raw) * len(SECONDARY_MULTIJAIL)
    assert {r.language for r in rows} == set(SECONDARY_MULTIJAIL)
    assert all(r.label == "harmful" and r.source == "multijail" for r in rows)
    first = next(r for r in rows if r.language == "bn")
    assert first.en_prompt == raw[0]["en"] and first.prompt == raw[0]["bn"]


def test_xstest_keeps_safe_only():
    rows = load_xstest(read_jsonl(FIX / "xstest_mini.jsonl"))
    assert len(rows) == 4
    assert all(r.label == "benign" and r.language == "en" and r.en_prompt == r.prompt for r in rows)
    assert all(r.id.startswith("xstest:") and r.category for r in rows)
