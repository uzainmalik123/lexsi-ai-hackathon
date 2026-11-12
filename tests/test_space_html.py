import json
from pathlib import Path

from ayasteer.publish.space import build_space
from ayasteer.schema import write_jsonl


def _judged(cond, lang, label, out, judge):
    return {"id": f"polyguard:1", "source": "polyguard", "language": lang, "label": label, "prompt": "native",
            "en_prompt": "english", "condition": cond, "strength": 1.0, "output": out, "finish_reason": "stop",
            "judge_label": judge}


def test_build_space_writes_index_and_data_json(tmp_path: Path):
    results = tmp_path / "results"
    write_jsonl(results / "judged/eval/base.jsonl", [_judged("base", "hi", "harmful", "sure, here", "compliance")])
    write_jsonl(results / "judged/eval/steer_en.jsonl", [_judged("steer_en", "hi", "harmful", "I can't", "refusal")])
    (results / "report").mkdir()
    (results / "report/report.md").write_text("## Headline\n\n| Language | x |\n|---|---|\n| Hindi | 1 |\n")
    (results / "report/gap.png").write_bytes(b"\x89PNG fake")

    out = tmp_path / "space"
    build_space(results, out)

    assert (out / "index.html").exists() and (out / "data.json").exists()
    data = json.loads((out / "data.json").read_text())
    assert set(data["conditions"]) == {"base", "steer_en"}
    assert data["languages"] == ["hi"]
    row = data["rows"][0]
    assert row["id"] == "polyguard:1" and row["language"] == "hi"
    assert row["outputs"]["base"]["judge_label"] == "compliance"
    assert row["outputs"]["steer_en"]["output"] == "I can't"
    html = (out / "index.html").read_text()
    assert "data.json" in html and "<select" in html and "Hindi" in html
    assert (out / "gap.png").exists()
