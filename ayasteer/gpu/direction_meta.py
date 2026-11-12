"""The JSON half of the steering-direction artifact (no torch, testable anywhere)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CONFIG_NAME = "steering_config.json"
TENSOR_NAME = "refusal_direction.safetensors"


def contrast_sha256(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r.get('id', '')}\t{r.get('prompt', '')}\n".encode("utf-8"))
    return h.hexdigest()


def save_meta(meta: dict, out_dir: Path, *, model_id: str, contrast_rows: list[dict], variant: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        **meta,
        "model_id": model_id,
        "variant": variant,
        "contrast_sha256": contrast_sha256(contrast_rows),
        "tensor_file": TENSOR_NAME,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / CONFIG_NAME).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg


def load_meta(dir: Path) -> dict:
    return json.loads((Path(dir) / CONFIG_NAME).read_text(encoding="utf-8"))
