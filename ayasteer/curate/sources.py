"""Load the three public sources into the shared Row schema.

Sources (all ungated on the HF Hub):
- ToxicityPrompts/PolyGuardPrompts (test): parallel harmful/benign prompts in 17 languages.
- DAMO-NLP-SG/MultiJail (train): human-translated harmful prompts in 10 languages.
- Paul/XSTest (train): English borderline prompts; we keep the 250 safe ones.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from ayasteer.schema import POLYGUARD_LANG, SECONDARY_MULTIJAIL, Row, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

POLYGUARD_HF = "ToxicityPrompts/PolyGuardPrompts"
MULTIJAIL_HF = "DAMO-NLP-SG/MultiJail"
XSTEST_HF = "Paul/XSTest"


def load_polyguard(rows: Iterable[dict]) -> list[Row]:
    """Non-adversarial PolyGuard rows in the headline languages, one Row per (id, language).

    An id is kept only if its English row exists (it supplies ``en_prompt``) and
    every kept row of that id is non-adversarial with a harm label.
    """
    by_id: dict[str, dict[str, dict]] = {}
    for r in rows:
        lang = POLYGUARD_LANG.get(r.get("language"))
        if lang is None:
            continue
        by_id.setdefault(str(r["id"]), {})[lang] = r

    out: list[Row] = []
    for pid, per_lang in by_id.items():
        en = per_lang.get("en")
        if en is None:
            continue
        if any(r.get("adversarial") or r.get("prompt_harm_label") not in ("harmful", "unharmful")
               for r in per_lang.values()):
            continue
        for lang, r in per_lang.items():
            out.append(Row(
                id=f"polyguard:{pid}",
                source="polyguard",
                language=lang,
                label="harmful" if r["prompt_harm_label"] == "harmful" else "benign",
                prompt=r["prompt"],
                en_prompt=en["prompt"],
                category=r.get("subcategory") or "unknown",
            ))
    return out


def load_multijail(rows: Iterable[dict]) -> list[Row]:
    """One harmful Row per (id, secondary language); English column is the parallel."""
    out: list[Row] = []
    for r in rows:
        for lang in SECONDARY_MULTIJAIL:
            text = r.get(lang)
            if not text:
                continue
            out.append(Row(
                id=f"multijail:{r['id']}",
                source="multijail",
                language=lang,
                label="harmful",
                prompt=text,
                en_prompt=r["en"],
                category=str(r.get("tags") or "unknown"),
            ))
    return out


def load_xstest(rows: Iterable[dict]) -> list[Row]:
    """XSTest safe prompts as English benign rows."""
    return [
        Row(
            id=f"xstest:{r['id']}",
            source="xstest",
            language="en",
            label="benign",
            prompt=r["prompt"],
            en_prompt=r["prompt"],
            category=r.get("type") or "unknown",
        )
        for r in rows
        if r.get("label") == "safe"
    ]


def fetch_sources(cache_dir: Path) -> dict[str, list[dict]]:
    """Download the three datasets (once) into ``cache_dir`` as JSONL and return their rows."""
    from datasets import load_dataset  # heavy import kept local

    cache_dir = Path(cache_dir)
    spec = {
        "polyguard": (POLYGUARD_HF, "test"),
        "multijail": (MULTIJAIL_HF, "train"),
        "xstest": (XSTEST_HF, "train"),
    }
    out: dict[str, list[dict]] = {}
    for name, (hf_id, split) in spec.items():
        path = cache_dir / f"{name}.jsonl"
        if not path.exists():
            log.info("downloading %s", hf_id)
            ds = load_dataset(hf_id, split=split)
            write_jsonl(path, (dict(r) for r in ds))
        out[name] = read_jsonl(path)
    return out
