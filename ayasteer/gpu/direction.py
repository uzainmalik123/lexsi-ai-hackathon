"""Refusal-direction extraction with SafeTune (Arditi et al.) and the tensor artifact."""
from __future__ import annotations

import contextlib
from pathlib import Path

import torch

from ayasteer.gpu.direction_meta import TENSOR_NAME, load_meta, resolve_pick_layer, save_meta

DEFAULT_PICK_LAYER = 24  # SafeTune docs: the unguarded sweep misfires on Tiny Aya; ~24 of 36 works


@contextlib.contextmanager
def detached_lm_head(model):
    """Swap the output head for an identity during activation extraction.

    Direction extraction only reads decoder-block hidden states, but a plain forward
    still materialises logits over Tiny Aya's 256k vocabulary (batch x seq x 256k in
    fp16 is several GB) and OOMs on a 16 GB card. The identity keeps the forward
    tiny; the model is restored on exit.
    """
    head = getattr(model, "lm_head", None)
    if head is None:
        yield
        return
    model.lm_head = torch.nn.Identity()
    try:
        yield
    finally:
        model.lm_head = head


def extract(model, tok, contrast_rows: list[dict], pick_layer: int | None = DEFAULT_PICK_LAYER):
    """Return (direction, layer, meta). ``pick_layer=None`` runs SafeTune's validation sweep.

    With ``pick_layer`` set no generation is needed, so the output head is detached for the
    extraction forward passes (see ``detached_lm_head``). The sweep needs logits, so it keeps the head.
    """
    from safetune.steer import extract_refusal_direction
    from safetune.interventions.steer.refusal_direction import RefusalDirectionConfig

    harmful = [r["prompt"] for r in contrast_rows if r["label"] == "harmful"]
    harmless = [r["prompt"] for r in contrast_rows if r["label"] == "benign"]
    n_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    pick_layer = resolve_pick_layer(pick_layer, n_layers) if n_layers else pick_layer
    cfg = RefusalDirectionConfig(pick_layer=pick_layer)
    ctx = detached_lm_head(model) if pick_layer is not None else contextlib.nullcontext()
    with ctx:
        direction, layer, _ = extract_refusal_direction(model, tok, harmful, harmless, cfg)
    meta = {
        "layer": int(layer),
        "n_harmful": len(harmful),
        "n_harmless": len(harmless),
        "pool_method": cfg.pool_method,
        "sweep_used": pick_layer is None,
        "hidden_dim": int(direction.shape[0]),
        "n_layers": n_layers,
    }
    return direction.detach().float().cpu(), int(layer), meta


def save_direction(direction: torch.Tensor, meta: dict, out_dir: Path, *, model_id: str,
                   contrast_rows: list[dict], variant: str) -> None:
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({"direction": direction.contiguous().float()}, str(out_dir / TENSOR_NAME))
    save_meta(meta, out_dir, model_id=model_id, contrast_rows=contrast_rows, variant=variant)


def load_direction(dir: Path) -> tuple[torch.Tensor, dict]:
    from safetensors.torch import load_file

    meta = load_meta(dir)
    return load_file(str(Path(dir) / TENSOR_NAME))["direction"], meta
