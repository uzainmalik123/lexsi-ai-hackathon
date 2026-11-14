"""Public copy of ``results/`` with harmful completions removed.

An output on a harmful prompt is kept only when the judge labelled it a refusal and the condition is
not an ablation; everything else keeps its row and labels but loses the text. Red-team rows keep their
text only when it is an explicit refusal. Rates stay reproducible from the labels.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ayasteer.schema import read_jsonl

REDACTED = "[redacted: harmful prompt, output not a refusal]"
SKIP_DIRS = {"space"}


def redact_row(row: dict, judge_label: str | None = None) -> dict:
    """Ablated conditions lose every harmful-prompt output: the judge calls some of those harmful
    answers refusals (soft openings cut off at 96 tokens), so its label is not trusted there."""
    label = row.get("judge_label", judge_label)
    ablated = str(row.get("condition", "")).startswith("ablate")
    if row.get("label") == "harmful" and "output" in row and (ablated or label != "refusal"):
        return {**row, "output": REDACTED, "redacted": True}
    return row


def redact_redteam(d: dict) -> dict:
    """AuditKIT's ``passed`` is canary-based (a complied bomb poem still passes), so keep only explicit refusals."""
    from ayasteer.gpu.layersweep import is_refusal

    rows = [r if is_refusal(r.get("output", "")) else {**r, "output": REDACTED, "redacted": True}
            for r in d.get("rows", [])]
    return {**d, "rows": rows}


def _labels_for(gen_path: Path, results: Path) -> dict:
    judged = results / "judged" / gen_path.relative_to(results / "gen")
    return {(r["id"], r["language"]): r.get("judge_label") for r in read_jsonl(judged)} if judged.exists() else {}


def build_public(results: Path, out: Path) -> dict:
    results, out = Path(results), Path(out)
    if out.exists():
        shutil.rmtree(out)
    counts = {"files": 0, "redacted": 0}
    for p in sorted(results.rglob("*")):
        rel = p.relative_to(results)
        if not p.is_file() or rel.parts[0] in SKIP_DIRS or p.suffix == ".bak":
            continue
        if rel.parts[0] == "directions" and rel.parent.name.endswith("_abl") and p.suffix == ".safetensors":
            continue  # the refusal-removal vectors: config and sweep table are published, the tensor is not
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        counts["files"] += 1
        if p.suffix == ".jsonl":
            labels = _labels_for(p, results) if rel.parts[0] == "gen" else {}
            rows = [redact_row(r, labels.get((r.get("id"), r.get("language")))) for r in read_jsonl(p)]
            counts["redacted"] += sum(bool(r.get("redacted")) for r in rows)
            dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        elif rel.parts[0] == "redteam" and p.suffix == ".json":
            d = redact_redteam(json.loads(p.read_text()))
            counts["redacted"] += sum(bool(r.get("redacted")) for r in d["rows"])
            dst.write_text(json.dumps(d, ensure_ascii=False, indent=1))
        elif p.name == "spotcheck.md":
            continue  # free-text dump of outputs; the redacted spotcheck.jsonl carries the same rows
        else:
            shutil.copy2(p, dst)
    return counts
