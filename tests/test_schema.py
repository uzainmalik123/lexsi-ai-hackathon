from pathlib import Path

import pytest

from ayasteer.schema import (
    HEADLINE,
    POLYGUARD_LANG,
    BELEBELE_CONFIG,
    LANG_NAME,
    Row,
    append_jsonl,
    read_jsonl,
    write_jsonl,
)


def _row(i: int) -> Row:
    return Row(
        id=f"polyguard:{i}",
        source="polyguard",
        language="hi",
        label="harmful",
        prompt=f"prompt {i}",
        en_prompt=f"english {i}",
        category="weapons",
        split="eval",
    )


def test_row_roundtrip_jsonl(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    rows = [_row(1), _row(2)]
    n = write_jsonl(path, [r.to_dict() for r in rows])
    assert n == 2
    back = read_jsonl(path)
    assert back == [r.to_dict() for r in rows]
    assert Row.from_dict(back[0]) == rows[0]


def test_append_jsonl_adds_without_truncating(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [_row(1).to_dict()])
    append_jsonl(path, [_row(2).to_dict()])
    assert [r["id"] for r in read_jsonl(path)] == ["polyguard:1", "polyguard:2"]


def test_write_jsonl_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "a" / "b" / "x.jsonl"
    write_jsonl(path, [])
    assert path.exists()


def test_label_must_be_harmful_or_benign():
    with pytest.raises(ValueError):
        Row(id="x", source="s", language="en", label="x", prompt="p", en_prompt="p")


def test_headline_order_fixed():
    assert HEADLINE == ["en", "hi", "ar", "zh", "es", "fr", "ru", "ja", "th"]
    assert set(POLYGUARD_LANG.values()) == set(HEADLINE)
    assert set(BELEBELE_CONFIG) == set(HEADLINE) | {"ur"}
    assert LANG_NAME["ur"] == "Urdu" and LANG_NAME["en"] == "English"
