"""CLI:
    python -m ayasteer.judge run        --in results/gen/eval/base.jsonl --out results/judged/eval/base.jsonl
    python -m ayasteer.judge run-dir    --in results/gen/dev --out results/judged/dev
    python -m ayasteer.judge agreement  --in results/gen/eval/base.jsonl --per-lang 30 --out results/judge_agreement.json
    python -m ayasteer.judge spotcheck  --in results/judged/eval/base.jsonl --langs en,ur --n 20 --out results/spotcheck.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ayasteer.judge.run import agreement_test, auditkit_run, judge_rows, sample_per_language
from ayasteer.schema import read_jsonl, write_jsonl


def _client(args):
    from ayasteer.llm.ollama import OllamaClient
    return OllamaClient(model=args.ollama_model)


def cmd_run(args):
    client = _client(args)
    rows = read_jsonl(Path(args.inp))
    n = judge_rows(rows, client, with_en_parallel=not args.no_en, out_path=Path(args.out))
    judged = read_jsonl(Path(args.out))
    ak = auditkit_run(judged, name=Path(args.out).stem, out_path=Path(args.out).with_suffix(".auditkit.json"))
    print(json.dumps({"judged_now": n, "total": len(judged), "auditkit": ak["stats"]}))


def cmd_run_dir(args):
    client = _client(args)
    for path in sorted(Path(args.inp).glob("*.jsonl")):
        out = Path(args.out) / path.name
        n = judge_rows(read_jsonl(path), client, with_en_parallel=not args.no_en, out_path=out)
        judged = read_jsonl(out)
        ak = auditkit_run(judged, name=out.stem, out_path=out.with_suffix(".auditkit.json"))
        print(json.dumps({"file": path.name, "judged_now": n, "total": len(judged), "auditkit": ak["stats"]}))


def cmd_agreement(args):
    client = _client(args)
    rows = sample_per_language(read_jsonl(Path(args.inp)), args.per_lang)
    base = Path(args.out).with_suffix("")
    judge_rows(rows, client, with_en_parallel=True, out_path=Path(f"{base}_with_en.jsonl"))
    judge_rows(rows, client, with_en_parallel=False, out_path=Path(f"{base}_without_en.jsonl"))
    agree = agreement_test(read_jsonl(Path(f"{base}_with_en.jsonl")), read_jsonl(Path(f"{base}_without_en.jsonl")))
    Path(args.out).write_text(json.dumps({"per_lang": args.per_lang, "agreement": agree}, indent=1))
    print(json.dumps(agree))


def cmd_spotcheck(args):
    rows = read_jsonl(Path(args.inp))
    langs = args.langs.split(",")
    picked = sample_per_language([r for r in rows if r["language"] in langs], args.n)
    out = [{"id": r["id"], "language": r["language"], "label": r["label"], "prompt": r["prompt"],
            "en_prompt": r["en_prompt"], "output": r["output"], "judge_label": r["judge_label"], "human": ""}
           for r in picked]
    write_jsonl(Path(args.out), out)
    print(f"wrote {len(out)} rows to {args.out}; fill the 'human' field with ok|wrong")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m ayasteer.judge")
    ap.add_argument("--ollama-model", default="qwen3.5-9b")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--in", dest="inp", required=True); p.add_argument("--out", required=True)
    p.add_argument("--no-en", action="store_true"); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("run-dir"); p.add_argument("--in", dest="inp", required=True); p.add_argument("--out", required=True)
    p.add_argument("--no-en", action="store_true"); p.set_defaults(fn=cmd_run_dir)
    p = sub.add_parser("agreement"); p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--per-lang", type=int, default=30); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_agreement)
    p = sub.add_parser("spotcheck"); p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--langs", default="en,ur"); p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", required=True); p.set_defaults(fn=cmd_spotcheck)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
