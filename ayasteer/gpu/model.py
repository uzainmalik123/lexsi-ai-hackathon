"""Tiny Aya loading, batched generation, and the SafeTune steering context."""
from __future__ import annotations

import contextlib
from typing import Any

import torch

DEFAULT_MODEL = "CohereLabs/tiny-aya-global"


def load_tiny_aya(model_id: str = DEFAULT_MODEL, device: str = "cuda"):
    """Load with SafeTune's loader (fp16 on CUDA) and a left-padding tokenizer."""
    from transformers import AutoTokenizer
    from safetune.runner.utils.model_utils import load_model

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(model_id, dtype=None, device=device)
    model.eval()
    return model, tok


def steering_context(model, direction: torch.Tensor | None, mode: str, strength: float):
    """``nullcontext`` for the base condition, else a SafeTune RefusalDirectionModel."""
    if direction is None or mode == "base":
        return contextlib.nullcontext()
    from safetune.steer import RefusalDirectionModel

    return RefusalDirectionModel(model, direction=direction, mode=mode, strength=strength)


@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], max_new_tokens: int = 96) -> list[tuple[str, str]]:
    """Greedy generation with the model's default chat template; returns (text, finish_reason)."""
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
             for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    results: list[tuple[str, str]] = []
    stop_ids = {tok.eos_token_id, tok.pad_token_id} | set(_extra_stop_ids(tok))
    for seq in gen:
        ids: list[int] = seq.tolist()
        hit_stop = any(i in stop_ids for i in ids)
        text = tok.decode(seq, skip_special_tokens=True).strip()
        if not text:
            reason = "empty"
        elif hit_stop:
            reason = "stop"
        else:
            reason = "length"
        results.append((text, reason))
    return results


def _extra_stop_ids(tok: Any) -> list[int]:
    ids = []
    for name in ("<|END_OF_TURN_TOKEN|>",):
        i = tok.convert_tokens_to_ids(name)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.append(i)
    return ids
