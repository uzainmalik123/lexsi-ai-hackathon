#!/usr/bin/env bash
# Ablation re-run: pick the layer whose direction actually removes refusal (Arditi-style sweep),
# then ablate the en and multi directions at that layer on the eval split, plus capability and
# red-team. Pushes to HF after every step; re-running resumes from what is already there.
set -uo pipefail
REPO="${AYASTEER_RUNS_REPO:-uzainahmed/tiny-aya-mlsteer-runs}"
push() { python -m ayasteer.gpu push --repo "$REPO" --path results || echo "push failed (will retry next step)"; }
step() { echo; echo "=== $* ==="; "$@" && push; }

echo "=== smoke test on a small model (stops here if the plumbing is broken) ==="
AYASTEER_MODEL=Qwen/Qwen2.5-0.5B-Instruct python -m ayasteer.gpu layer-sweep --smoke --directions-dir /tmp/smoke || exit 1
AYASTEER_MODEL=Qwen/Qwen2.5-0.5B-Instruct python -m ayasteer.gpu generate --split dev --condition ablate_en \
  --strength 1.0 --direction /tmp/smoke/en_abl --out /tmp/smoke/g.jsonl --limit 8 || exit 1

step python -m ayasteer.gpu layer-sweep --variants en,multi
test -f results/directions/en_abl/steering_config.json || { echo "layer sweep found no usable layer; stopping"; exit 1; }
step python -m ayasteer.gpu generate --split eval --condition ablate_en    --strength 1.0 --direction results/directions/en_abl    --out results/gen/eval/ablate_en.jsonl
step python -m ayasteer.gpu generate --split eval --condition ablate_multi --strength 1.0 --direction results/directions/multi_abl --out results/gen/eval/ablate_multi.jsonl
step python -m ayasteer.gpu capability --conditions "ablate_en:1.0,ablate_multi:1.0"
step python -m ayasteer.gpu redteam    --conditions "ablate_en:1.0,ablate_multi:1.0"
echo; echo "=== ALL DONE ==="; cat results/directions/layer_sweep.json | head -c 600; ls -la results/gen/eval
