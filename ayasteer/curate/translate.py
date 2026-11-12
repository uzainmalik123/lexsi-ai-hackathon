"""Machine translation of benchmark rows with the local Ollama model.

Only two things are translated (see the spec): the XSTest borderline-benign
prompts into every non-English headline language plus Urdu, and the English
PolyGuard eval rows into Urdu. Every translated row carries ``translated_by``.
"""
from __future__ import annotations

import logging
import random
from dataclasses import replace
from pathlib import Path

from ayasteer.schema import HEADLINE, LANG_NAME, TRANSLATED, Row, append_jsonl, read_jsonl

log = logging.getLogger(__name__)

SYSTEM = (
    "You are a professional translator. Translate the user's text faithfully, preserving "
    "its intent, tone, and level of detail, even if the request is unusual or sensitive; "
    "you are translating, not answering. Output only the translation, nothing else."
)


def _target_langs() -> list[str]:
    return [l for l in HEADLINE if l != "en"] + TRANSLATED


def translate_rows(rows: list[Row], target_lang: str, client) -> list[Row]:
    name = LANG_NAME[target_lang]
    out: list[Row] = []
    for r in rows:
        text = client.chat(SYSTEM, f"Translate into {name}:\n\n{r.prompt}", max_tokens=400).strip()
        out.append(replace(
            r,
            id=f"{r.id}:{target_lang}",
            language=target_lang,
            prompt=text,
            en_prompt=r.prompt,
            translated_by=f"ollama/{client.model}",
        ))
    return out


def back_translate_check(rows: list[Row], client, fraction: float = 0.1, seed: int = 20261112) -> list[dict]:
    rng = random.Random(seed)
    k = max(1, round(len(rows) * fraction)) if rows else 0
    sample = rng.sample(rows, k)
    out = []
    for r in sample:
        back = client.chat(SYSTEM, f"Translate into English:\n\n{r.prompt}", max_tokens=400).strip()
        out.append({"id": r.id, "original": r.en_prompt, "back": back})
    return out


def build_translations(splits: dict[str, list[Row]], client, out_dir: Path,
                       langs: list[str] | None = None) -> list[Row]:
    """Translate and persist per-language JSONL; resumable by row id."""
    out_dir = Path(out_dir)
    langs = langs or _target_langs()
    xstest_en = [r for r in splits["eval"] if r.source == "xstest" and r.language == "en"]
    polyguard_en = [r for r in splits["eval"] if r.source == "polyguard" and r.language == "en"]

    result: list[Row] = []
    for lang in langs:
        path = out_dir / f"{lang}.jsonl"
        done = {d["id"]: Row.from_dict(d) for d in read_jsonl(path)}
        sources = list(xstest_en) + (list(polyguard_en) if lang in TRANSLATED else [])
        todo = [r for r in sources if f"{r.id}:{lang}" not in done]
        log.info("translating %d rows into %s (%d already done)", len(todo), lang, len(done))
        for r in todo:
            new = translate_rows([r], lang, client)[0]
            append_jsonl(path, [new.to_dict()])
            done[new.id] = new
        result.extend(done[f"{r.id}:{lang}"] for r in sources)
    return result
