"""CLI:
    python -m ayasteer.report select --judged-dev results/judged/dev --out results/report/selected_strengths.json
    python -m ayasteer.report build  --judged results/judged --capability results/capability --redteam results/redteam \
                                     --gen results/gen --out results/report [--roi-gpu 0.35 --roi-in 3 --roi-out 15]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from ayasteer.report.figures import before_after_figure, gap_figure, sweep_figure
from ayasteer.report.metrics import gap_vs_english, mean_over, rates
from ayasteer.report.roi import DEFAULT_CLOSED_IN_USD_PER_M, DEFAULT_CLOSED_OUT_USD_PER_M, DEFAULT_GPU_USD_PER_HOUR, roi
from ayasteer.report.select_strength import select_strength
from ayasteer.report.tables import capability_table, headline_table, redteam_table, secondary_table
from ayasteer.schema import read_jsonl

log = logging.getLogger("ayasteer.report")


def _rates_with_mean(path: Path) -> dict:
    r = rates(read_jsonl(path))
    r["__mean__"] = mean_over(r)
    return r


def _load_sweep(judged_dev: Path) -> tuple[dict, dict]:
    base = _rates_with_mean(judged_dev / "base.jsonl")
    sweep = {}
    for p in sorted(judged_dev.glob("steer_*_s*.jsonl")):
        m = re.match(r"steer_(en|multi)_s([\d.]+)\.jsonl", p.name)
        if m:
            sweep[(m.group(1), float(m.group(2)))] = _rates_with_mean(p)
    return sweep, base


def cmd_select(args):
    sweep, base = _load_sweep(Path(args.judged_dev))
    chosen = select_strength(sweep, base, args.max_or_increase, args.max_incoherent)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    table = {f"{v}:{s:g}": r["__mean__"] for (v, s), r in sweep.items()}
    out.write_text(json.dumps({"selected": chosen, "rule": {"max_or_increase": args.max_or_increase,
                               "max_incoherent": args.max_incoherent}, "base_mean": base["__mean__"], "sweep_mean": table}, indent=1))
    sweep_figure(sweep, base, out.parent / "sweep.png")
    print(json.dumps(chosen))


def cmd_build(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    judged = Path(args.judged) / "eval"
    results = {}
    for cond in ("base", "steer_en", "steer_multi", "ablate"):
        p = judged / f"{cond}.jsonl"
        if p.exists():
            results[cond] = rates(read_jsonl(p))
    md = ["## Headline (PolyGuard + translated XSTest, judge = Ollama qwen3.5-9b)", "", headline_table(results), "",
          "## Secondary languages", "", secondary_table(results), ""]
    cap = {}
    for p in sorted(Path(args.capability).glob("*.json")) if Path(args.capability).exists() else []:
        d = json.loads(p.read_text()); cap[d["condition"]] = d
    if cap:
        md += ["## Capability retention (Belebele, 4-way MC, AuditKIT exact-letter)", "", capability_table(cap), ""]
    rt = {}
    for p in sorted(Path(args.redteam).glob("*.json")) if Path(args.redteam).exists() else []:
        d = json.loads(p.read_text()); rt[d["condition"]] = d
    if rt:
        md += ["## AuditKIT built-in red-team probes", "", redteam_table(rt), ""]
    gaps = gap_vs_english(results["base"]) if "base" in results else {}
    figs = []
    if "base" in results:
        figs.append(gap_figure(results["base"], out / "gap.png"))
        figs.append(before_after_figure(results, "harmful_refusal", out / "before_after_refusal.png",
                                        "Refusal rate on harmful prompts per language, by condition"))
        figs.append(before_after_figure(results, "harmful_compliance", out / "before_after_compliance.png",
                                        "Harmful compliance per language, by condition"))
        figs.append(before_after_figure(results, "over_refusal", out / "before_after_overrefusal.png",
                                        "Over-refusal on benign prompts per language, by condition"))
    meta = sorted(Path(args.gen).glob("eval/*.meta.json")) if Path(args.gen).exists() else []
    tps = [json.loads(p.read_text()).get("tokens_per_s", 0) for p in meta]
    tps = [t for t in tps if t]
    r = roi(tokens_per_s=sum(tps) / len(tps) if tps else float("nan"), gpu_usd_per_hour=args.roi_gpu,
            requests=args.roi_requests, in_tokens=200, out_tokens=96,
            closed_in_usd_per_m=args.roi_in, closed_out_usd_per_m=args.roi_out)
    md += ["## ROI (per month)", "", f"```json\n{json.dumps(r, indent=1)}\n```", ""]
    (out / "results.json").write_text(json.dumps({"rates": results, "gap_vs_english": gaps, "capability": cap,
                                                  "redteam": {k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in rt.items()},
                                                  "roi": r}, indent=1, default=str))
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md[:4]))
    print("figures:", [str(f) for f in figs])


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(prog="python -m ayasteer.report")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select"); p.add_argument("--judged-dev", default="results/judged/dev")
    p.add_argument("--out", default="results/report/selected_strengths.json")
    p.add_argument("--max-or-increase", type=float, default=0.10); p.add_argument("--max-incoherent", type=float, default=0.10)
    p.set_defaults(fn=cmd_select)
    p = sub.add_parser("build"); p.add_argument("--judged", default="results/judged")
    p.add_argument("--capability", default="results/capability"); p.add_argument("--redteam", default="results/redteam")
    p.add_argument("--gen", default="results/gen"); p.add_argument("--out", default="results/report")
    p.add_argument("--roi-gpu", type=float, default=DEFAULT_GPU_USD_PER_HOUR)
    p.add_argument("--roi-in", type=float, default=DEFAULT_CLOSED_IN_USD_PER_M)
    p.add_argument("--roi-out", type=float, default=DEFAULT_CLOSED_OUT_USD_PER_M)
    p.add_argument("--roi-requests", type=int, default=1_000_000)
    p.set_defaults(fn=cmd_build)
    args = ap.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
