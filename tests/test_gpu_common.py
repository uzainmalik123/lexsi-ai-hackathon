from pathlib import Path

from ayasteer.gpu.common import CONDITIONS, batched, make_result_row, pending_rows
from ayasteer.schema import write_jsonl


def test_batched_sizes():
    assert list(batched(list(range(7)), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(batched([], 3)) == []


def test_result_row_has_all_fields():
    row = {"id": "polyguard:1", "source": "polyguard", "language": "hi", "label": "harmful",
           "prompt": "p", "en_prompt": "e", "category": "c", "split": "eval", "translated_by": ""}
    out = make_result_row(row, condition="steer_en", strength=2.0, output="x", finish_reason="stop")
    assert out == {
        "id": "polyguard:1", "source": "polyguard", "language": "hi", "label": "harmful",
        "prompt": "p", "en_prompt": "e", "condition": "steer_en", "strength": 2.0,
        "output": "x", "finish_reason": "stop",
    }


def test_pending_rows_skips_written_ids(tmp_path: Path):
    rows = [{"id": "a", "language": "en"}, {"id": "b", "language": "en"}, {"id": "a", "language": "hi"}]
    out = tmp_path / "gen.jsonl"
    assert pending_rows(rows, out) == rows  # file absent -> everything pending
    write_jsonl(out, [{"id": "a", "language": "en", "output": "done"}])
    assert pending_rows(rows, out) == rows[1:]


def test_conditions_fixed():
    assert CONDITIONS == ("base", "steer_en", "steer_multi", "ablate", "ablate_en", "ablate_multi")


def test_balanced_limit_takes_half_of_each_label():
    from ayasteer.gpu.common import balanced_limit
    rows = [{"id": f"h{i}", "label": "harmful"} for i in range(5)] + [{"id": f"b{i}", "label": "benign"} for i in range(5)]
    out = balanced_limit(rows, 4)
    assert [r["label"] for r in out] == ["harmful", "harmful", "benign", "benign"]
    assert balanced_limit(rows, 0) == rows
    assert len(balanced_limit(rows, 7)) == 7
