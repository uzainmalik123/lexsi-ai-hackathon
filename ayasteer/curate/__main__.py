"""CLI: python -m ayasteer.curate --out data/benchmark --raw data/raw [--no-translate]"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ayasteer.curate.build import SEED, build


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the multilingual safety benchmark.")
    ap.add_argument("--out", default="data/benchmark")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-translate", action="store_true", help="skip the Ollama translation stage")
    ap.add_argument("--ollama-model", default="qwen3.5-9b")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    client = None
    if not args.no_translate:
        from ayasteer.llm.ollama import OllamaClient
        client = OllamaClient(model=args.ollama_model)
    manifest = build(Path(args.out), Path(args.raw), client, seed=args.seed)
    print(json.dumps({"counts": manifest["counts"], "stages": manifest["stages"]}, indent=1))


if __name__ == "__main__":
    main()
