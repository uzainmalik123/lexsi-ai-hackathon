#!/usr/bin/env bash
# Day-3 GPU run: final eval conditions, capability, red-team. Pushes to HF after every step,
# and every step resumes from what is already on HF, so re-running after a crash only redoes
# the unfinished part.
set -uo pipefail
S_EN="${S_EN:-0.2}"; S_MULTI="${S_MULTI:-0.1}"
REPO="${AYASTEER_RUNS_REPO:-uzainahmed/tiny-aya-mlsteer-runs}"
push() { python -m ayasteer.gpu push --repo "$REPO" --path results || echo "push failed (will retry next step)"; }
step() { echo; echo "=== $* ==="; "$@" && push; }

step python -m ayasteer.gpu generate --split eval --condition steer_en    --strength "$S_EN"    --direction results/directions/en    --out results/gen/eval/steer_en.jsonl
step python -m ayasteer.gpu generate --split eval --condition steer_multi --strength "$S_MULTI" --direction results/directions/multi --out results/gen/eval/steer_multi.jsonl
step python -m ayasteer.gpu generate --split eval --condition ablate      --strength 1.0        --direction results/directions/en    --out results/gen/eval/ablate.jsonl
step python -m ayasteer.gpu capability --conditions "base,steer_en:$S_EN,steer_multi:$S_MULTI,ablate:1.0"
step python -m ayasteer.gpu redteam    --conditions "base,steer_en:$S_EN,steer_multi:$S_MULTI,ablate:1.0"
echo; echo "=== ALL DONE ==="; ls -la results/gen/eval results/capability results/redteam
