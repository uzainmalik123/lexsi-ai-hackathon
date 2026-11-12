import contextlib
import json
from pathlib import Path

from ayasteer.gpu.generate import run_condition
from ayasteer.schema import read_jsonl


class FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _rows(n):
    return [{"id": f"x:{i}", "source": "polyguard", "language": "en", "label": "harmful",
             "prompt": f"p{i}", "en_prompt": f"p{i}"} for i in range(n)]


def test_run_condition_resumes_after_partial_file(tmp_path: Path):
    out = tmp_path / "gen.jsonl"
    calls: list[list[str]] = []

    def gen_fn(model, tok, prompts, max_new_tokens=96):
        calls.append(prompts)
        if len(calls) == 2:
            raise RuntimeError("session died")
        return [(f"out {p}", "stop") for p in prompts]

    rows = _rows(5)
    try:
        run_condition(None, FakeTok(), rows, "base", 0.0, None, out, batch_size=2,
                      gen_fn=gen_fn, ctx_fn=lambda *a: contextlib.nullcontext())
    except RuntimeError:
        pass
    assert [r["id"] for r in read_jsonl(out)] == ["x:0", "x:1"]  # first batch persisted

    calls.clear()
    stats = run_condition(None, FakeTok(), rows, "base", 0.0, None, out, batch_size=2,
                          gen_fn=lambda m, t, p, max_new_tokens=96: [(f"out {x}", "stop") for x in p],
                          ctx_fn=lambda *a: contextlib.nullcontext())
    ids = [r["id"] for r in read_jsonl(out)]
    assert ids == ["x:0", "x:1", "x:2", "x:3", "x:4"] and len(set(ids)) == 5
    assert stats["n"] == 3 and stats["n_total"] == 5
    meta = json.loads(out.with_suffix(".jsonl.meta.json").read_text())
    assert meta["condition"] == "base" and len(meta["runs"]) == 1  # the crashed run wrote no meta


def test_empty_output_row_is_written_with_finish_reason(tmp_path: Path):
    out = tmp_path / "gen.jsonl"
    run_condition(None, FakeTok(), _rows(1), "steer_en", 2.0, "dir", out, batch_size=1,
                  gen_fn=lambda m, t, p, max_new_tokens=96: [("", "empty")],
                  ctx_fn=lambda *a: contextlib.nullcontext())
    row = read_jsonl(out)[0]
    assert row["output"] == "" and row["finish_reason"] == "empty"
    assert row["condition"] == "steer_en" and row["strength"] == 2.0
