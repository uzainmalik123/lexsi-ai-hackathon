from pathlib import Path

from ayasteer.judge.detectors import detector_labels
from ayasteer.judge.run import agreement_test, judge_rows
from ayasteer.judge.rubric import LABELS, build_user_prompt, parse_choice
from ayasteer.schema import read_jsonl


class FakeClient:
    model = "qwen3.5-9b"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def chat(self, system, user, **kw):
        self.calls.append((system, user))
        return self.replies.pop(0)


def _row(i, output="I can't help with that.", lang="hi", label="harmful"):
    return {"id": f"polyguard:{i}", "source": "polyguard", "language": lang, "label": label,
            "prompt": f"native {i}", "en_prompt": f"english {i}", "condition": "base", "strength": 0.0,
            "output": output, "finish_reason": "stop"}


def test_parse_choice_each_label():
    for label in LABELS:
        assert parse_choice(f"Some reasoning.\nCHOICE: {label}") == label
        assert parse_choice(f"choice: {label.upper()}") == label


def test_parse_choice_unknown_on_garbage():
    assert parse_choice("") == "unknown"
    assert parse_choice("The model refused politely.") == "unknown"
    assert parse_choice("CHOICE: maybe") == "unknown"


def test_empty_output_is_marked_in_prompt():
    row = _row(1, output="")
    user = build_user_prompt(row, with_en_parallel=True)
    assert "[EMPTY OUTPUT]" in user
    assert "english 1" in user and "native 1" in user and "Hindi" in user


def test_prompt_without_english_parallel_omits_it():
    user = build_user_prompt(_row(1), with_en_parallel=False)
    assert "english 1" not in user and "native 1" in user


def test_judge_rows_resumes_and_retries_unknown_once(tmp_path: Path):
    out = tmp_path / "judged.jsonl"
    rows = [_row(1), _row(2), _row(3)]
    # row1 ok; row2 garbage then ok on retry; row3 garbage twice -> unknown
    client = FakeClient(["CHOICE: refusal", "???", "CHOICE: compliance", "???", "???"])
    n = judge_rows(rows, client, with_en_parallel=True, out_path=out)
    assert n == 3 and len(client.calls) == 5
    judged = read_jsonl(out)
    assert [r["judge_label"] for r in judged] == ["refusal", "compliance", "unknown"]
    assert all(r["judge_with_en"] is True and "judge_raw" in r for r in judged)

    # resume: nothing left to do, client never called
    client2 = FakeClient([])
    assert judge_rows(rows, client2, with_en_parallel=True, out_path=out) == 0
    assert client2.calls == [] and len(read_jsonl(out)) == 3


def test_detector_labels_on_known_refusal():
    d = detector_labels("I'm sorry, but I can't help with that request.")
    assert d == {"auditkit_refusal": True, "arditi_refusal": True}
    assert detector_labels("Sure! Here is how you do it.") == {"auditkit_refusal": False, "arditi_refusal": False}


def test_agreement_test_per_language():
    a = [dict(_row(1, lang="hi"), judge_label="refusal"), dict(_row(2, lang="hi"), judge_label="compliance"),
         dict(_row(3, lang="ar"), judge_label="refusal")]
    b = [dict(_row(1, lang="hi"), judge_label="refusal"), dict(_row(2, lang="hi"), judge_label="refusal"),
         dict(_row(3, lang="ar"), judge_label="refusal")]
    assert agreement_test(a, b) == {"hi": 0.5, "ar": 1.0}
