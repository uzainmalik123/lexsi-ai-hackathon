"""Run one condition over a split, appending per batch so a dead session can resume."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ayasteer.gpu.common import batched, make_result_row, pending_rows
from ayasteer.schema import append_jsonl

log = logging.getLogger(__name__)

MODE_FOR_CONDITION = {"base": "base", "steer_en": "steer", "steer_multi": "steer", "ablate": "ablate"}


def run_condition(model, tok, rows: list[dict], condition: str, strength: float, direction, out_path: Path,
                  batch_size: int = 16, max_new_tokens: int = 96, gen_fn=None, ctx_fn=None) -> dict:
    """``gen_fn`` / ``ctx_fn`` default to the torch implementations; tests inject fakes."""
    if gen_fn is None or ctx_fn is None:
        from ayasteer.gpu.model import generate_batch, steering_context
        gen_fn = gen_fn or generate_batch
        ctx_fn = ctx_fn or steering_context
    out_path = Path(out_path)
    todo = pending_rows(rows, out_path)
    log.info("%s s=%g: %d rows to generate (%d already done)", condition, strength, len(todo), len(rows) - len(todo))
    mode = MODE_FOR_CONDITION[condition]
    t0 = time.time()
    n_tokens = 0
    with ctx_fn(model, direction, mode, strength):
        for batch in batched(todo, batch_size):
            outs = gen_fn(model, tok, [r["prompt"] for r in batch], max_new_tokens=max_new_tokens)
            n_tokens += sum(len(tok(text, add_special_tokens=False)["input_ids"]) for text, _ in outs)
            append_jsonl(out_path, (make_result_row(r, condition, strength, text, reason)
                                    for r, (text, reason) in zip(batch, outs)))
    seconds = time.time() - t0
    stats = {
        "condition": condition, "strength": strength, "n": len(todo), "n_total": len(rows),
        "seconds": round(seconds, 1), "generated_tokens": n_tokens,
        "tokens_per_s": round(n_tokens / seconds, 2) if seconds > 0 else 0.0,
        "batch_size": batch_size, "max_new_tokens": max_new_tokens,
        "device": str(getattr(model, "device", "?")),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    prev = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    stats["runs"] = prev.get("runs", []) + [{"n": len(todo), "seconds": stats["seconds"], "tokens_per_s": stats["tokens_per_s"]}]
    meta_path.write_text(json.dumps(stats, indent=1))
    return stats
