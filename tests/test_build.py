import json
import shutil
from pathlib import Path

from ayasteer.curate.build import build
from ayasteer.schema import read_jsonl

FIX = Path(__file__).parent / "fixtures"


def _raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(FIX / "polyguard_mini.jsonl", raw / "polyguard.jsonl")
    shutil.copy(FIX / "multijail_mini.jsonl", raw / "multijail.jsonl")
    # inject an exact duplicate of the first safe XSTest prompt under a new id
    xs = read_jsonl(FIX / "xstest_mini.jsonl")
    first_safe = next(r for r in xs if r["label"] == "safe")
    dup = dict(first_safe, id=9999)
    with (raw / "xstest.jsonl").open("w", encoding="utf-8") as fh:
        for r in xs + [dup]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return raw


def test_build_on_fixtures_writes_all_files_and_dedups(tmp_path: Path):
    out = tmp_path / "benchmark"
    manifest = build(out_dir=out, raw_dir=_raw_dir(tmp_path), client=None, seed=1)

    for name in ("eval.jsonl", "dev.jsonl", "contrast_en.jsonl", "contrast_multi.jsonl",
                 "manifest.json", "lexsi_provenance.json"):
        assert (out / name).exists(), name

    rows = read_jsonl(out / "eval.jsonl")
    prompts = [(r["prompt"], r["language"]) for r in rows]
    assert len(prompts) == len(set(prompts)), "duplicate prompt survived"
    xs = read_jsonl(FIX / "xstest_mini.jsonl")
    first_safe_id = f"xstest:{next(r for r in xs if r['label'] == 'safe')['id']}"
    survivors = [r["id"] for r in rows if r["id"] in (first_safe_id, "xstest:9999")]
    assert len(survivors) == 1, survivors

    stages = manifest["stages"]
    dedup = [s for s in stages if "dedup" in s["name"].lower()]
    assert dedup and any(s["in"] > s["out"] for s in dedup)
    assert manifest["counts"]["eval"] == len(rows)
    assert manifest["translated"] is False
    assert manifest["seed"] == 1
