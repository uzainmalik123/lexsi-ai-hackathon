"""Pure helpers for the GPU side. No torch import so they are testable anywhere."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ayasteer.schema import read_jsonl

CONDITIONS = ("base", "steer_en", "steer_multi", "ablate")

RESULT_FIELDS = ("id", "source", "language", "label", "prompt", "en_prompt")


def batched(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def make_result_row(row: dict, condition: str, strength: float, output: str, finish_reason: str) -> dict:
    out = {k: row[k] for k in RESULT_FIELDS}
    out.update(condition=condition, strength=strength, output=output, finish_reason=finish_reason)
    return out


def pending_rows(rows: list[dict], out_path: Path) -> list[dict]:
    done = {(r["id"], r["language"]) for r in read_jsonl(Path(out_path))}
    return [r for r in rows if (r["id"], r["language"]) not in done]
