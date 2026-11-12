# Multilingual refusal steering on Tiny Aya Global

Does Cohere's Tiny Aya Global refuse harmful requests equally in every language it speaks, and can one
refusal direction, found with SafeTune and added back at inference time, close the gaps without
retraining or breaking benign behaviour? This repo builds a parallel safety benchmark in 14 languages
with CuratorKIT, measures the model as shipped, steers it with SafeTune, and scores everything with
AuditKIT plus a local LLM judge.

Lexsi AI Safety and Alignment Hackathon 2026, **Track 2** (AI Red Teaming, Safety and Alignment),
problem statements **2a #4** (multilingual and code-mixed safety) and **2b #8** (inference-time safety
steering). Solo entry.

## Results

_Filled in from `results/report/report.md` once the final runs are judged._

## Where the Lexsi repos are used

| repo | what it does here | code |
|---|---|---|
| [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) | ingests the three sources as `prompt_only` samples, exact + MinHash dedup, provenance manifest, checksums, dataset card | `ayasteer/curate/build.py` (`_run_curatorkit`) |
| [SafeTune](https://github.com/Lexsi-Labs/SafeTune) | loads the model (`load_model`), extracts the refusal direction (`extract_refusal_direction`, Arditi et al.), applies it at inference (`RefusalDirectionModel`, modes `steer` and `ablate`) | `ayasteer/gpu/model.py`, `ayasteer/gpu/direction.py` |
| [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) | Belebele capability eval via `ak.evaluate` with a `FunctionScorer`; built-in `RedTeamRunner` probes; `RefusalDetector` as a secondary metric; every judged file wrapped in an `ak.evaluate(model="precomputed")` RunResult for a fingerprint | `ayasteer/gpu/capability.py`, `ayasteer/gpu/redteam.py`, `ayasteer/judge/detectors.py`, `ayasteer/judge/run.py` |

Base model: `CohereLabs/tiny-aya-global` (3.35B, cohere2, 36 layers), fp16, default chat template with
Cohere's own safety preamble. Judge and translator: `qwen3.5-9b` through local Ollama (allowed by the
hackathon FAQ as a supplementary model).

## Repo map

```
ayasteer/
  schema.py            Row schema, JSONL io, language tables
  curate/              sources.py (PolyGuard, MultiJail, XSTest) · splits.py · translate.py (Ollama) · build.py (CuratorKIT)
  gpu/                 model.py · direction.py · generate.py · capability.py · redteam.py · push.py   [runs on Kaggle/Colab]
  judge/               rubric.py (frozen) · run.py (resumable Ollama judge + AuditKIT wrap) · detectors.py
  report/              metrics.py · select_strength.py · tables.py · figures.py · roi.py
  publish/             hf.py (datasets, models, collections, space) · space.py (static replay page)
data/benchmark/        the curated benchmark (committed; a few MB)
notebooks/KAGGLE.md    the GPU cells, in order
cards/                 HF dataset and model cards
docs/superpowers/      design spec and implementation plan
tests/                 pytest suite (no GPU, no network)
```

## Reproduce

Two machines: a laptop without a GPU for curation, judging and reporting; a free Kaggle or Colab GPU
for everything that touches Tiny Aya. Seeds are fixed (`20261112`) everywhere.

**Laptop (Python 3.12, uv, Ollama with `qwen3.5-9b` pulled):**

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[local]"
pytest                                                     # 54 tests, ~2 s

# 1. benchmark (downloads the three sources; translation ≈ 100 min on an RTX 2060 Super)
python -m ayasteer.curate --out data/benchmark --raw data/raw
```

**GPU notebook:** paste the cells from `notebooks/KAGGLE.md` in order. Cells 4 and 5 are smoke tests
(a tiny open model, then Tiny Aya on 16 rows). Cell 6 extracts both directions, runs the base
condition on `eval` and the strength sweep on `dev`, and pushes `results/` to the HF dataset repo.

**Laptop again:**

```bash
# 2. pull the GPU outputs, judge the base run and the sweep (≈ 0.4 s per row on the 2060 Super with Ollama on the GPU)
python -c "from huggingface_hub import snapshot_download; snapshot_download('uzainahmed/tiny-aya-mlsteer-runs', repo_type='dataset', local_dir='.')"
python -m ayasteer.judge run     --in results/gen/eval/base.jsonl --out results/judged/eval/base.jsonl
python -m ayasteer.judge run-dir --in results/gen/dev --out results/judged/dev
python -m ayasteer.judge agreement --in results/gen/eval/base.jsonl --per-lang 30 --out results/judge_agreement.json
python -m ayasteer.judge spotcheck --in results/judged/eval/base.jsonl --langs en,ur --n 20 --out results/spotcheck.jsonl

# 3. pick the steering strengths with the rule from the spec
python -m ayasteer.report select --judged-dev results/judged/dev --out results/report/selected_strengths.json
```

**GPU notebook, cell 7** with the selected strengths: final conditions, Belebele, red-team probes, push.

```bash
# 4. judge the final conditions, build the report, publish
python -m ayasteer.judge run-dir --in results/gen/eval --out results/judged/eval
python -m ayasteer.report build
python -m ayasteer.publish all
```

Runtimes measured on this project are recorded in `results/gen/**/*.meta.json` (tokens/s on the GPU)
and in the write-up.

## Limitations

- The judge is a local 9B model, not a human. We report its agreement with and without the English
  parallel prompt per language, and a 20-row human check in English and Urdu.
- XSTest and the Urdu slices are machine-translated.
- Steering strength is chosen on a separate dev split, but by the same judge.
- MultiJail-only languages have no benign prompts, so no over-refusal number there.

## Licences

Code: MIT. Lexsi libraries: LSAL (source-available, non-commercial). Data cards list each source's licence.
