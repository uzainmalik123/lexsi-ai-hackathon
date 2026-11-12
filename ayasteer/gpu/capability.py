"""Capability retention: Belebele reading comprehension scored through AuditKIT."""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from ayasteer.schema import BELEBELE_CONFIG

log = logging.getLogger(__name__)

LETTERS = "ABCD"
_LETTER_RE = re.compile(r"\b([A-Da-d])\b")


def format_belebele(row: dict) -> tuple[str, str]:
    options = "\n".join(f"{L}. {row[f'mc_answer{i}']}" for i, L in enumerate(LETTERS, start=1))
    prompt = (f"{row['flores_passage']}\n\nQuestion: {row['question']}\n{options}\n\n"
              "Answer with a single letter.")
    target = LETTERS[int(row["correct_answer_num"]) - 1]
    return prompt, target


def normalize_letter(pred: str) -> str:
    m = _LETTER_RE.search(pred or "")
    return m.group(1).upper() if m else ""


def score_exact_letter(target: str, pred: str) -> float:
    return 1.0 if normalize_letter(pred) == target else 0.0


def belebele_samples(lang: str, n: int = 50, seed: int = 20261112) -> list:
    from datasets import load_dataset
    import auditkit as ak

    ds = load_dataset("facebook/belebele", BELEBELE_CONFIG[lang], split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    samples = []
    for i in idx[:n]:
        prompt, target = format_belebele(ds[i])
        samples.append(ak.Sample(input=prompt, target=target, id=f"belebele:{lang}:{ds[i]['link']}:{ds[i]['question_number']}",
                                 metadata={"language": lang}))
    return samples


def run_capability(model, tok, langs: list[str], condition: str, strength: float, direction,
                   out_path: Path, n_per_lang: int = 50) -> dict:
    import auditkit as ak
    from ayasteer.gpu.generate import MODE_FOR_CONDITION
    from ayasteer.gpu.model import generate_batch, steering_context

    def callable_model(prompts: list[str]) -> list[str]:
        outs = []
        for i in range(0, len(prompts), 16):
            outs.extend(t for t, _ in generate_batch(model, tok, prompts[i:i + 16], max_new_tokens=4))
        return outs

    from auditkit.api import FunctionScorer
    from auditkit.types import Direction

    scorer = FunctionScorer(lambda sample, output: score_exact_letter(sample.target, output),
                            direction=Direction.MAXIMIZE, name="letter_exact")

    results: dict[str, float] = {}
    runs: dict[str, dict] = {}
    with steering_context(model, direction, MODE_FOR_CONDITION[condition], strength):
        for lang in langs:
            samples = belebele_samples(lang, n=n_per_lang)
            run = ak.evaluate(samples, model=callable_model, scorers=[scorer], cache=False,
                              experiment_name=f"belebele_{condition}_s{strength:g}", tags=[lang])
            acc = _mean(run.stats["letter_exact"])
            results[lang] = round(acc, 4)
            runs[lang] = {"fingerprint": run.fingerprint, "run_id": run.run_id, "summary": run.summary()}
            log.info("belebele %s %s s=%g: %.3f", lang, condition, strength, acc)
    out = {"condition": condition, "strength": strength, "n_per_lang": n_per_lang, "accuracy": results,
           "auditkit_runs": runs}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    return out


def _mean(stat) -> float:
    """AuditKIT's per-metric stats object (or dict) -> mean."""
    if isinstance(stat, dict):
        return float(stat.get("mean", stat.get("value")))
    return float(getattr(stat, "mean", getattr(stat, "value", float("nan"))))
