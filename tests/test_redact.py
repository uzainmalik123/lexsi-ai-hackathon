import json

from ayasteer.publish.redact import REDACTED, build_public, redact_redteam, redact_row


def _r(label, judge=None, out="text", i="1"):
    row = {"id": i, "language": "en", "label": label, "output": out}
    if judge:
        row["judge_label"] = judge
    return row


def test_redact_row_keeps_refusals_and_benign():
    assert redact_row(_r("harmful", "refusal"))["output"] == "text"
    assert redact_row(_r("benign", "compliance"))["output"] == "text"


def test_redact_row_removes_non_refusals_and_unjudged():
    for judge in ("compliance", "partial", "incoherent", None):
        r = redact_row(_r("harmful", judge))
        assert r["output"] == REDACTED and r["redacted"]


def test_redact_row_uses_external_label_for_gen_rows():
    assert redact_row(_r("harmful"), judge_label="refusal")["output"] == "text"


def test_redact_redteam_keeps_passed_only():
    d = redact_redteam({"rows": [{"output": "a", "passed": True}, {"output": "b", "passed": False}]})
    assert [r["output"] for r in d["rows"]] == ["a", REDACTED]


def test_build_public_joins_gen_with_judged(tmp_path):
    res = tmp_path / "results"
    (res / "gen/eval").mkdir(parents=True); (res / "judged/eval").mkdir(parents=True)
    rows = [_r("harmful", out="no", i="1"), _r("harmful", out="bad", i="2")]
    (res / "gen/eval/x.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    judged = [{**rows[0], "judge_label": "refusal"}, {**rows[1], "judge_label": "compliance"}]
    (res / "judged/eval/x.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judged))
    (res / "space").mkdir(); (res / "space/index.html").write_text("x")
    counts = build_public(res, tmp_path / "pub")
    gen = [json.loads(l) for l in (tmp_path / "pub/gen/eval/x.jsonl").read_text().splitlines()]
    assert [g["output"] for g in gen] == ["no", REDACTED]
    assert counts["redacted"] == 2 and not (tmp_path / "pub/space").exists()
