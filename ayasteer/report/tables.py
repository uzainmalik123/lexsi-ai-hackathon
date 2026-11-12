"""Markdown tables for the README and the HF Article."""
from __future__ import annotations

import math

from ayasteer.schema import HEADLINE, LANG_NAME, SECONDARY_MULTIJAIL, TRANSLATED

COND_LABEL = {"base": "base", "steer_en": "steer (en dir)", "steer_multi": "steer (multi dir)", "ablate": "ablated"}


def _pct(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"


def headline_table(results: dict[str, dict[str, dict]], conds=("base", "steer_en", "steer_multi", "ablate")) -> str:
    """results[condition][lang] -> rates dict."""
    conds = [c for c in conds if c in results]
    head = "| Language | " + " | ".join(f"harmful compliance ({COND_LABEL[c]})" for c in conds) + \
           " | gap vs en (base) | " + " | ".join(f"over-refusal ({COND_LABEL[c]})" for c in conds) + " |"
    sep = "|" + "---|" * (2 + 2 * len(conds))
    lines = [head, sep]
    en_base = results.get("base", {}).get("en", {}).get("harmful_compliance", float("nan"))
    for lang in HEADLINE:
        if lang not in results.get("base", {}):
            continue
        hc = [_pct(results[c].get(lang, {}).get("harmful_compliance")) for c in conds]
        orr = [_pct(results[c].get(lang, {}).get("over_refusal")) for c in conds]
        gap = results["base"][lang]["harmful_compliance"] - en_base
        gap_s = "–" if math.isnan(gap) else f"{100 * gap:+.0f} pts"
        lines.append(f"| {LANG_NAME[lang]} | " + " | ".join(hc) + f" | {gap_s} | " + " | ".join(orr) + " |")
    return "\n".join(lines)


def secondary_table(results: dict[str, dict[str, dict]], conds=("base", "steer_en", "steer_multi", "ablate")) -> str:
    conds = [c for c in conds if c in results]
    lines = ["| Language | source | " + " | ".join(f"harmful compliance ({COND_LABEL[c]})" for c in conds) +
             " | " + " | ".join(f"over-refusal ({COND_LABEL[c]})" for c in conds) + " |",
             "|" + "---|" * (2 + 2 * len(conds))]
    for lang in SECONDARY_MULTIJAIL + TRANSLATED:
        if lang not in results.get("base", {}):
            continue
        src = "MultiJail (human-translated)" if lang in SECONDARY_MULTIJAIL else "machine-translated (Ollama)"
        hc = [_pct(results[c].get(lang, {}).get("harmful_compliance")) for c in conds]
        orr = [_pct(results[c].get(lang, {}).get("over_refusal")) for c in conds]
        lines.append(f"| {LANG_NAME[lang]} | {src} | " + " | ".join(hc) + " | " + " | ".join(orr) + " |")
    return "\n".join(lines)


def capability_table(cap: dict[str, dict]) -> str:
    """cap[condition] = {"accuracy": {lang: acc}}."""
    conds = list(cap)
    langs = [l for l in HEADLINE + TRANSLATED if any(l in cap[c]["accuracy"] for c in conds)]
    lines = ["| Belebele accuracy | " + " | ".join(COND_LABEL.get(c, c) for c in conds) + " |", "|" + "---|" * (1 + len(conds))]
    for lang in langs:
        lines.append(f"| {LANG_NAME[lang]} | " + " | ".join(_pct(cap[c]["accuracy"].get(lang)) for c in conds) + " |")
    means = [sum(cap[c]["accuracy"].values()) / len(cap[c]["accuracy"]) for c in conds]
    lines.append("| **mean** | " + " | ".join(f"**{_pct(m)}**" for m in means) + " |")
    return "\n".join(lines)


def redteam_table(rt: dict[str, dict]) -> str:
    conds = list(rt)
    probes = sorted({p for c in conds for p in rt[c]["per_probe"]})
    lines = ["| AuditKIT probe (pass rate) | " + " | ".join(COND_LABEL.get(c, c) for c in conds) + " |",
             "|" + "---|" * (1 + len(conds))]
    for p in probes:
        cells = []
        for c in conds:
            d = rt[c]["per_probe"].get(p)
            cells.append(_pct(d["passed"] / max(1, d["total"] - d["invalid"])) if d else "–")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    lines.append("| **all** | " + " | ".join(_pct(rt[c]["pass_rate"]) for c in conds) + " |")
    return "\n".join(lines)
