"""CLI for the GPU side.

    python -m ayasteer.gpu direction --variant en --out results/directions/en [--model ID] [--sweep]
    python -m ayasteer.gpu generate  --split eval --condition steer_en --strength 1 --direction DIR --out F
    python -m ayasteer.gpu sweep     --split dev --variants en,multi --strengths 0.5,1,2,4
    python -m ayasteer.gpu capability --conditions base,steer_en:1.0 --langs en,hi
    python -m ayasteer.gpu redteam    --conditions base,steer_en:1.0
    python -m ayasteer.gpu push      --repo USER/REPO --path results
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from ayasteer.gpu.common import balanced_limit
from ayasteer.schema import read_jsonl

log = logging.getLogger("ayasteer.gpu")
BENCH = Path("data/benchmark")


def _add_model_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=os.environ.get("AYASTEER_MODEL", "CohereLabs/tiny-aya-global"))


def cmd_direction(args: argparse.Namespace) -> None:
    from ayasteer.gpu.direction import extract, save_direction
    from ayasteer.gpu.model import load_tiny_aya

    rows = balanced_limit(read_jsonl(BENCH / f"contrast_{args.variant}.jsonl"), args.limit)
    model, tok = load_tiny_aya(args.model)
    direction, layer, meta = extract(model, tok, rows, pick_layer=None if args.sweep else args.pick_layer)
    save_direction(direction, meta, Path(args.out), model_id=args.model, contrast_rows=rows, variant=args.variant)
    print(json.dumps({"variant": args.variant, "layer": layer, **meta, "out": args.out}))


def cmd_generate(args: argparse.Namespace) -> None:
    from ayasteer.gpu.generate import run_condition
    from ayasteer.gpu.model import load_tiny_aya

    rows = balanced_limit(read_jsonl(BENCH / f"{args.split}.jsonl"), args.limit)
    model, tok = load_tiny_aya(args.model)
    direction = None
    if args.condition != "base":
        from ayasteer.gpu.direction import load_direction
        direction, _ = load_direction(Path(args.direction))
    stats = run_condition(model, tok, rows, args.condition, args.strength, direction, Path(args.out),
                          batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    print(json.dumps(stats))


def cmd_sweep(args: argparse.Namespace) -> None:
    from ayasteer.gpu.direction import load_direction
    from ayasteer.gpu.generate import run_condition
    from ayasteer.gpu.model import load_tiny_aya

    rows = balanced_limit(read_jsonl(BENCH / f"{args.split}.jsonl"), args.limit)
    model, tok = load_tiny_aya(args.model)
    out_dir = Path(args.out_dir) / args.split
    print(json.dumps(run_condition(model, tok, rows, "base", 0.0, None, out_dir / "base.jsonl",
                                   batch_size=args.batch_size)))
    for variant in args.variants.split(","):
        direction, _ = load_direction(Path(args.directions_dir) / variant)
        for s in (float(x) for x in args.strengths.split(",")):
            out = out_dir / f"steer_{variant}_s{s:g}.jsonl"
            print(json.dumps({"variant": variant, "strength": s,
                              **run_condition(model, tok, rows, f"steer_{variant}", s, direction, out,
                                              batch_size=args.batch_size)}))


def _parse_conditions(spec: str) -> list[tuple[str, float]]:
    out = []
    for item in spec.split(","):
        name, _, s = item.partition(":")
        out.append((name, float(s) if s else 0.0))
    return out


def cmd_capability(args: argparse.Namespace) -> None:
    from ayasteer.gpu.capability import run_capability
    from ayasteer.gpu.direction import load_direction
    from ayasteer.gpu.model import load_tiny_aya

    model, tok = load_tiny_aya(args.model)
    dirs = {v: load_direction(Path(args.directions_dir) / v)[0] for v in ("en", "multi")
            if (Path(args.directions_dir) / v).exists()}
    for cond, s in _parse_conditions(args.conditions):
        direction = None if cond == "base" else dirs["en" if cond in ("ablate", "steer_en") else "multi"]
        out = Path(args.out_dir) / f"{cond}_s{s:g}.json"
        print(json.dumps(run_capability(model, tok, args.langs.split(","), cond, s, direction, out,
                                        n_per_lang=args.n_per_lang)))


def cmd_redteam(args: argparse.Namespace) -> None:
    from ayasteer.gpu.direction import load_direction
    from ayasteer.gpu.model import load_tiny_aya
    from ayasteer.gpu.redteam import run_redteam

    model, tok = load_tiny_aya(args.model)
    dirs = {v: load_direction(Path(args.directions_dir) / v)[0] for v in ("en", "multi")
            if (Path(args.directions_dir) / v).exists()}
    for cond, s in _parse_conditions(args.conditions):
        direction = None if cond == "base" else dirs["en" if cond in ("ablate", "steer_en") else "multi"]
        out = Path(args.out_dir) / f"{cond}_s{s:g}.json"
        print(json.dumps(run_redteam(model, tok, cond, s, direction, out)))


def cmd_push(args: argparse.Namespace) -> None:
    from ayasteer.gpu.push import push_results

    print(push_results(args.repo, Path(args.path), token=os.environ.get("HF_TOKEN")))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m ayasteer.gpu")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("direction"); _add_model_arg(p)
    p.add_argument("--variant", choices=["en", "multi"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pick-layer", type=int, default=24)
    p.add_argument("--sweep", action="store_true", help="run SafeTune's layer-selection sweep instead")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_direction)

    p = sub.add_parser("generate"); _add_model_arg(p)
    p.add_argument("--split", default="eval")
    p.add_argument("--condition", choices=["base", "steer_en", "steer_multi", "ablate"], required=True)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--direction", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("sweep"); _add_model_arg(p)
    p.add_argument("--split", default="dev")
    p.add_argument("--variants", default="en,multi")
    p.add_argument("--strengths", default="0.5,1,2,4")
    p.add_argument("--directions-dir", default="results/directions")
    p.add_argument("--out-dir", default="results/gen")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("capability"); _add_model_arg(p)
    p.add_argument("--conditions", required=True, help="e.g. base,steer_en:2.0,steer_multi:1.0,ablate:1.0")
    p.add_argument("--langs", default="en,hi,ar,zh,es,fr,ru,ja,th,ur")
    p.add_argument("--n-per-lang", type=int, default=50)
    p.add_argument("--directions-dir", default="results/directions")
    p.add_argument("--out-dir", default="results/capability")
    p.set_defaults(fn=cmd_capability)

    p = sub.add_parser("redteam"); _add_model_arg(p)
    p.add_argument("--conditions", required=True)
    p.add_argument("--directions-dir", default="results/directions")
    p.add_argument("--out-dir", default="results/redteam")
    p.set_defaults(fn=cmd_redteam)

    p = sub.add_parser("push")
    p.add_argument("--repo", required=True)
    p.add_argument("--path", default="results")
    p.set_defaults(fn=cmd_push)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
