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

Full write-up: [`writeup/article.md`](writeup/article.md). Full tables: `results/report/report.md`.

- **No large multilingual gap in the base model.** Tiny Aya refuses 83–93% of harmful prompts in the nine
  headline languages and 73–80% in Bengali, Swahili, Vietnamese and Korean; harmful compliance is 0–5%
  everywhere.
- **Positive steering (layer 24) buys little.** +2.4 points of refusal for +4.3 points of over-refusal
  (English direction), +1.7 for +2.0 (multilingual direction). Belebele accuracy unchanged.
- **One direction carries refusal in every language.** Ablating the layer-19 direction, found by an
  Arditi-style layer sweep, cuts explicit English refusals from 82% to 10% (en direction) and 7% (multi
  direction), and lowers judged refusal by 12–35 points in all 14 languages, including five that neither
  direction was extracted from. Capability (Belebele, 76% mean) is unchanged. The English and multilingual
  directions have cosine 0.96.
- **Measurement caveats we found:** the 9B judge counts soft redirects (and harmful answers cut off at 96
  tokens) as refusals, so it understates the ablation effect; AuditKIT's built-in red-team detectors are
  canary-based and score every condition at 83%.

| mean over 9 headline languages | base | steer en | steer multi | ablate en (L19) | ablate multi (L19) |
|---|---|---|---|---|---|
| refusal on harmful (judge) | 87.2% | 89.6% | 88.9% | 67.6% | 61.9% |
| explicit refusals, English (substring) | 82% | 87% | 85% | 10% | 7% |
| over-refusal on benign | 14.1% | 18.4% | 16.1% | 14.0% | 13.9% |
| Belebele accuracy (10 languages) | 76% | 76% | 76% | 76% | 76% |

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
pytest                                                     # 68 tests, ~2 s

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

**GPU notebooks** (Kaggle: import, GPU T4 + Internet + `HF_TOKEN` secret, Save & Run All):
`notebooks/kaggle_final.ipynb` runs the steered and layer-24 ablated eval conditions, Belebele and the
red-team probes; `notebooks/kaggle_ablate.ipynb` runs the layer sweep (`python -m ayasteer.gpu layer-sweep`)
and the layer-19 ablations. Both push to HF after every step and resume from there.

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
  parallel prompt per language (96% mean) and a 40-row human check in English and Urdu (87.5% on refusal
  vs not). It is lenient on ablated outputs; see the write-up.
- Generation is capped at 96 new tokens, and n = 60 harmful prompts per language.
- XSTest and the Urdu slices are machine-translated.
- Steering strength is chosen on a separate dev split, but by the same judge.
- The public runs dataset redacts the text of every harmful-prompt output the judge did not label a
  refusal (labels kept), so no harmful completions are redistributed.
- MultiJail-only languages have no benign prompts, so no over-refusal number there.

## Licences

Code: MIT. Lexsi libraries: LSAL (source-available, non-commercial). Data cards list each source's licence.
