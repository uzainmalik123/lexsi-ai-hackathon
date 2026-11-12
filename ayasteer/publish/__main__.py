"""CLI:
    python -m ayasteer.publish benchmark | runs | models | space | collections | all
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ayasteer.judge.rubric import RUBRIC, SYSTEM
from ayasteer.publish.hf import REPOS, create_collection, publish_dataset, publish_model, publish_space
from ayasteer.publish.space import build_space

CARDS = Path("cards")


def _manifest_counts() -> str:
    p = Path("data/benchmark/manifest.json")
    if not p.exists():
        return "(manifest missing)"
    m = json.loads(p.read_text())
    return ", ".join(f"{k}: {v}" for k, v in m["counts"].items())


def do_benchmark(args) -> str:
    return publish_dataset(REPOS["benchmark"], Path("data/benchmark"), CARDS / "dataset_benchmark.md",
                           counts=_manifest_counts(), rubric=RUBRIC, rubric_system=SYSTEM)


def do_runs(args) -> str:
    return publish_dataset(REPOS["runs"], Path("results"), CARDS / "dataset_runs.md", rubric=RUBRIC, rubric_system=SYSTEM)


def do_models(args) -> list[str]:
    urls = []
    for v in ("en", "multi"):
        d = Path("results/directions") / v
        if not d.exists():
            logging.warning("no direction at %s, skipping", d)
            continue
        cfg = json.loads((d / "steering_config.json").read_text())
        urls.append(publish_model(REPOS[f"direction_{v}"], d, CARDS / "model_direction.md", variant=v,
                                  layer=cfg.get("layer"), n_harmful=cfg.get("n_harmful"), n_harmless=cfg.get("n_harmless"),
                                  contrast_sha256=cfg.get("contrast_sha256"), created=cfg.get("created")))
    return urls


def do_space(args) -> str:
    out = build_space(Path("results"), Path("results/space"))
    return publish_space(REPOS["space"], out)


def do_collections(args) -> list[str]:
    models = [("model", REPOS[f"direction_{v}"]) for v in ("en", "multi") if (Path("results/directions") / v).exists()]
    datasets = [("dataset", REPOS["benchmark"]), ("dataset", REPOS["runs"])]
    return [
        create_collection("Tiny Aya multilingual refusal steering: models",
                          models + [("space", REPOS["space"])], "Refusal-direction steering artifacts for CohereLabs/tiny-aya-global (Lexsi hackathon 2026)"),
        create_collection("Tiny Aya multilingual refusal steering: datasets", datasets,
                          "Multilingual safety benchmark, generations, judge outputs, AuditKIT runs (Lexsi hackathon 2026)"),
    ]


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(prog="python -m ayasteer.publish")
    ap.add_argument("what", choices=["benchmark", "runs", "models", "space", "collections", "all"])
    args = ap.parse_args()
    steps = {"benchmark": do_benchmark, "runs": do_runs, "models": do_models, "space": do_space, "collections": do_collections}
    todo = list(steps) if args.what == "all" else [args.what]
    for name in todo:
        print(name, "->", steps[name](args))


if __name__ == "__main__":
    main()
