"""Row schema, JSONL I/O, and language tables shared by every layer."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

LABELS = ("harmful", "benign")

# Headline languages, in the order used in every table and figure.
HEADLINE = ["en", "hi", "ar", "zh", "es", "fr", "ru", "ja", "th"]
# MultiJail-only languages: harmful prompts only, secondary table.
SECONDARY_MULTIJAIL = ["bn", "sw", "vi", "ko"]
# Machine-translated languages (human spot-check: Urdu).
TRANSLATED = ["ur"]

POLYGUARD_LANG = {
    "English": "en",
    "Hindi": "hi",
    "Arabic": "ar",
    "Chinese": "zh",
    "Spanish": "es",
    "French": "fr",
    "Russian": "ru",
    "Japanese": "ja",
    "Thai": "th",
}

BELEBELE_CONFIG = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ar": "arb_Arab",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "th": "tha_Thai",
    "ur": "urd_Arab",
}

LANG_NAME = {
    "en": "English",
    "hi": "Hindi",
    "ar": "Arabic",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "ru": "Russian",
    "ja": "Japanese",
    "th": "Thai",
    "bn": "Bengali",
    "sw": "Swahili",
    "vi": "Vietnamese",
    "ko": "Korean",
    "ur": "Urdu",
}


@dataclass(frozen=True)
class Row:
    id: str
    source: str
    language: str
    label: str
    prompt: str
    en_prompt: str
    category: str = ""
    split: str = ""
    translated_by: str = ""

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {self.label!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Row":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _dump(fh, rows: Iterable[dict]) -> int:
    n = 0
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    return n


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        return _dump(fh, rows)


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        return _dump(fh, rows)
