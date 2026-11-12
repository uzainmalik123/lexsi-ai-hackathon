"""Assemble the benchmark: sources -> splits -> translations -> CuratorKIT -> data/benchmark/.

CuratorKIT runs the dedup / cleaning / provenance stages over every prompt
(as ``prompt_only`` samples whose ``id`` is our row id). Rows CuratorKIT
rejects are dropped from our split files, and its ``lexsi_provenance.json``,
``dataset_card`` and checksums are kept beside them.
"""
from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ayasteer.curate.sources import fetch_sources, load_multijail, load_polyguard, load_xstest
from ayasteer.curate.splits import SPLITS, assign_splits
from ayasteer.curate.translate import back_translate_check, build_translations
from ayasteer.schema import Row, write_jsonl

log = logging.getLogger(__name__)

SEED = 20261112


def _run_curatorkit(rows: list[Row], out_dir: Path) -> tuple[set[str], list[dict]]:
    """Run CuratorKIT over all prompts; return surviving row ids and stage counts."""
    from curatorkit import Curator, CuratorConfig

    ck_dir = out_dir / "curatorkit"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "prompts.jsonl"
        write_jsonl(src, ({"id": r.id, "language": r.language, "prompt": r.prompt} for r in rows))
        cfg = CuratorConfig(
            dataset=str(src),
            format="prompt_only",
            field_mapping={"instruction": "prompt"},
            min_tokens=1,
            max_tokens=4096,
            dedup="minhash",
            clean=False,  # prompts must reach the model byte-for-byte as sourced
            output_dir=str(ck_dir),
            name="ayasteer_benchmark",
        )
        result = Curator(cfg).run()

    kept = {s.id for s in result.passed}
    stages = [{"name": name,
               "in": c.get("input_count", c.get("output_count", 0)),
               "out": c.get("output_count", 0),
               "rejected": c.get("rejected_count", 0)}
              for name, c in (result.stage_counts or {}).items()]
    for fname in ("lexsi_provenance.json", "manifest.json", "checksums.txt", "dataset_card.md", "README.md"):
        p = ck_dir / fname
        if p.exists() and fname != "manifest.json":
            (out_dir / fname).write_bytes(p.read_bytes())
    return kept, stages


def _dedupe_within(rows: list[Row]) -> list[Row]:
    """Keep the first row per (prompt, language); CuratorKIT's id-level verdict is applied first."""
    seen: set[tuple[str, str]] = set()
    out: list[Row] = []
    for r in rows:
        key = (r.prompt, r.language)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build(out_dir: Path, raw_dir: Path, client, seed: int = SEED) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = fetch_sources(Path(raw_dir))
    splits = assign_splits(load_polyguard(raw["polyguard"]), load_multijail(raw["multijail"]),
                           load_xstest(raw["xstest"]), seed=seed)

    translated = client is not None
    back_sample: list[dict] = []
    if translated:
        new = build_translations(splits, client, out_dir / "translations")
        splits["eval"] = splits["eval"] + [replace(r, split="eval") for r in new]
        back_sample = back_translate_check(new, client, fraction=0.1, seed=seed)
        (out_dir / "back_translation_sample.json").write_text(
            json.dumps(back_sample, ensure_ascii=False, indent=1), encoding="utf-8")

    all_rows = [r for name in SPLITS for r in splits[name]]
    kept, stages = _run_curatorkit(all_rows, out_dir)
    dropped = [r.id for r in all_rows if r.id not in kept]
    log.info("CuratorKIT kept %d of %d prompts", len(kept), len(all_rows))

    counts: dict[str, int] = {}
    for name in SPLITS:
        rows = _dedupe_within([r for r in splits[name] if r.id in kept])
        counts[name] = write_jsonl(out_dir / f"{name}.jsonl", (r.to_dict() for r in rows))

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "translated": translated,
        "translator": getattr(client, "model", None) if translated else None,
        "counts": counts,
        "stages": stages,
        "dropped_by_curatorkit": dropped,
        "back_translation_sample_size": len(back_sample),
        "sources": {
            "polyguard": "ToxicityPrompts/PolyGuardPrompts (test)",
            "multijail": "DAMO-NLP-SG/MultiJail (train)",
            "xstest": "Paul/XSTest (train, safe only)",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest
