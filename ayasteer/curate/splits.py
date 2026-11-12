"""Assign PolyGuard ids to contrast / dev / eval splits, shared across languages."""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import replace

from ayasteer.schema import HEADLINE, Row

log = logging.getLogger(__name__)

N_EVAL = 60
N_DEV = 10
N_CONTRAST = 64
N_XSTEST = 60
N_MULTIJAIL = 60

SPLITS = ("eval", "dev", "contrast_en", "contrast_multi")


def _stratified_pick(ids_by_cat: dict[str, list[str]], n: int, rng: random.Random) -> list[str]:
    """Round-robin over shuffled categories until n ids are picked (or ids run out)."""
    groups = [rng.sample(v, len(v)) for v in ids_by_cat.values()]
    rng.shuffle(groups)
    picked: list[str] = []
    while len(picked) < n and any(groups):
        for g in groups:
            if g and len(picked) < n:
                picked.append(g.pop())
    return picked


def _sizes(available: int) -> tuple[int, int, int]:
    """(contrast, dev, eval) counts for one label; shrinks proportionally when short."""
    want = N_CONTRAST + N_DEV + N_EVAL
    if available >= want:
        return N_CONTRAST, N_DEV, N_EVAL
    log.warning("only %d ids available for a label, wanted %d; shrinking splits", available, want)
    if available < 3:
        raise ValueError("need at least 3 ids per label to form contrast/dev/eval")
    dev = max(1, round(available * N_DEV / want))
    contrast = max(1, round(available * N_CONTRAST / want))
    ev = available - dev - contrast
    if ev < 1:
        contrast -= 1 - ev
        ev = 1
    return contrast, dev, ev


def assign_splits(polyguard: list[Row], multijail: list[Row], xstest: list[Row],
                  seed: int = 20261112) -> dict[str, list[Row]]:
    rng = random.Random(seed)
    out: dict[str, list[Row]] = {s: [] for s in SPLITS}

    # --- PolyGuard: pick base ids per label from the English rows, apply to all languages ---
    by_id: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in polyguard:
        by_id[r.id][r.language] = r
    complete = {pid: d for pid, d in by_id.items() if set(d) >= set(HEADLINE)}
    if len(complete) < len(by_id):
        log.warning("dropping %d PolyGuard ids missing a headline language", len(by_id) - len(complete))

    for label in ("harmful", "benign"):
        cats: dict[str, list[str]] = defaultdict(list)
        for pid, d in complete.items():
            if d["en"].label == label:
                cats[d["en"].category].append(pid)
        n_contrast, n_dev, n_eval = _sizes(sum(len(v) for v in cats.values()))
        contrast_ids = _stratified_pick(cats, n_contrast, rng)
        _remove(cats, contrast_ids)
        dev_ids = _stratified_pick(cats, n_dev, rng)
        _remove(cats, dev_ids)
        eval_ids = _stratified_pick(cats, n_eval, rng)

        non_en = [l for l in HEADLINE if l != "en"]
        for i, pid in enumerate(contrast_ids):
            out["contrast_en"].append(replace(complete[pid]["en"], split="contrast_en"))
            lang = non_en[i % len(non_en)]
            out["contrast_multi"].append(replace(complete[pid][lang], split="contrast_multi"))
        for pid in dev_ids:
            out["dev"].extend(replace(complete[pid][l], split="dev") for l in HEADLINE)
        for pid in eval_ids:
            out["eval"].extend(replace(complete[pid][l], split="eval") for l in HEADLINE)

    # --- XSTest: stratified by type ---
    xcats: dict[str, list[str]] = defaultdict(list)
    xrows = {r.id: r for r in xstest}
    for r in xstest:
        xcats[r.category].append(r.id)
    for xid in _stratified_pick(xcats, N_XSTEST, rng):
        out["eval"].append(replace(xrows[xid], split="eval"))

    # --- MultiJail: same ids across its languages ---
    mj_by_id: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in multijail:
        mj_by_id[r.id][r.language] = r
    mj_ids = sorted(mj_by_id)
    rng.shuffle(mj_ids)
    for mid in mj_ids[:N_MULTIJAIL]:
        out["eval"].extend(replace(r, split="eval") for r in mj_by_id[mid].values())

    _check_unique(out)
    return out


def _remove(cats: dict[str, list[str]], ids: list[str]) -> None:
    gone = set(ids)
    for k in list(cats):
        cats[k] = [i for i in cats[k] if i not in gone]
        if not cats[k]:
            del cats[k]


def _check_unique(splits: dict[str, list[Row]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for name, rows in splits.items():
        for r in rows:
            key = (r.id, r.language)
            if key in seen:
                raise ValueError(f"duplicate (id, language) {key}: in {seen[key]} and {name}")
            seen[key] = name
    base: dict[str, str] = {}
    for name, rows in splits.items():
        for r in rows:
            prev = base.get(r.id)
            if prev is not None and prev != name and {prev, name} != {"contrast_en", "contrast_multi"}:
                raise ValueError(f"id {r.id} appears in both {prev} and {name}")
            base[r.id] = name
