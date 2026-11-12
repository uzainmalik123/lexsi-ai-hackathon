from pathlib import Path

import pytest

from ayasteer.curate.sources import load_multijail, load_polyguard, load_xstest
from ayasteer.curate.splits import assign_splits
from ayasteer.schema import HEADLINE, Row, read_jsonl

FIX = Path(__file__).parent / "fixtures"


def _splits():
    pg = load_polyguard(read_jsonl(FIX / "polyguard_mini.jsonl"))
    mj = load_multijail(read_jsonl(FIX / "multijail_mini.jsonl"))
    xs = load_xstest(read_jsonl(FIX / "xstest_mini.jsonl"))
    return assign_splits(pg, mj, xs, seed=1)


def _base_ids(rows, lang):
    return {r.id for r in rows if r.language == lang and r.source == "polyguard"}


def test_splits_have_expected_keys_and_split_field():
    s = _splits()
    assert set(s) == {"eval", "dev", "contrast_en", "contrast_multi"}
    for name, rows in s.items():
        assert rows and all(r.split == name for r in rows)


def test_splits_share_ids_across_languages():
    s = _splits()
    ref = _base_ids(s["eval"], "en")
    assert ref
    for lang in HEADLINE:
        assert _base_ids(s["eval"], lang) == ref


def test_splits_are_disjoint():
    s = _splits()
    seen: dict[str, str] = {}
    for name, rows in s.items():
        for r in rows:
            key = (r.id, r.language)
            assert key not in seen, f"{key} in both {seen[key]} and {name}"
            seen[key] = name
    # base PolyGuard ids never cross splits either
    base = {k: {r.id for r in v if r.source == "polyguard"} for k, v in s.items()}
    assert base["eval"].isdisjoint(base["dev"])
    assert base["eval"].isdisjoint(base["contrast_en"])
    assert base["dev"].isdisjoint(base["contrast_en"])


def test_contrast_multi_uses_non_english_languages_only():
    s = _splits()
    langs = {r.language for r in s["contrast_multi"]}
    assert "en" not in langs and langs <= set(HEADLINE)
    assert {r.id for r in s["contrast_multi"]} == {r.id for r in s["contrast_en"]}
    assert all(r.language == "en" for r in s["contrast_en"])


def test_eval_includes_multijail_and_xstest():
    s = _splits()
    assert any(r.source == "multijail" for r in s["eval"])
    assert any(r.source == "xstest" for r in s["eval"])
    assert not any(r.source != "polyguard" for r in s["dev"] + s["contrast_en"] + s["contrast_multi"])


def test_duplicate_id_raises():
    pg = load_polyguard(read_jsonl(FIX / "polyguard_mini.jsonl"))
    dup = Row(id=pg[0].id, source="xstest", language="en", label="benign", prompt="x", en_prompt="x")
    with pytest.raises(ValueError):
        assign_splits(pg, [], [dup], seed=1)
