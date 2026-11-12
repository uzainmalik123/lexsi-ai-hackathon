# Kaggle (or Colab) cells

Create a notebook with a GPU (T4 is enough), turn Internet on, and add your Hugging Face token as a
secret named `HF_TOKEN` (Add-ons → Secrets). Paste these cells in order. Every command is resumable:
re-running a cell after a dead session continues where it stopped.

**Cell 1 — Python version check.** SafeTune needs 3.12+. If this prints 3.11 or lower, open the same
notebook on Google Colab instead (Colab runs 3.12) and continue from Cell 2 there.

```python
!python --version
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Cell 2 — clone and install.**

```python
!git clone https://github.com/uzainmalik123/lexsi-ai-hackathon
%cd lexsi-ai-hackathon
!pip install -q -e ".[gpu]"
```

**Cell 3 — HF token into the environment.** (On Colab: `from google.colab import userdata` and
`userdata.get("HF_TOKEN")` instead.)

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

**Cell 4 — smoke test on a tiny open model** (two minutes; proves the plumbing before touching Tiny Aya).

```python
!AYASTEER_MODEL=Qwen/Qwen2.5-0.5B-Instruct python -m ayasteer.gpu direction --variant en --out /tmp/d --limit 16
!AYASTEER_MODEL=Qwen/Qwen2.5-0.5B-Instruct python -m ayasteer.gpu generate --split dev --condition steer_en --strength 1 --direction /tmp/d --out /tmp/g.jsonl --limit 8
!head -c 1500 /tmp/g.jsonl; echo; cat /tmp/g.jsonl.meta.json
```

Expected: 8 rows in `/tmp/g.jsonl`, each with `finish_reason`, and `tokens_per_s > 0` in the meta file.

**Cell 5 — Tiny Aya smoke** (loads the real model; a few minutes).

```python
!python -m ayasteer.gpu direction --variant en --out results/directions/en --limit 16
!python -m ayasteer.gpu generate --split dev --condition base --out /tmp/base16.jsonl --limit 16
!python -m ayasteer.gpu generate --split dev --condition steer_en --strength 1 --direction results/directions/en --out /tmp/steer16.jsonl --limit 16
!python - <<'EOF'
import json
b=[json.loads(l) for l in open('/tmp/base16.jsonl')]; s=[json.loads(l) for l in open('/tmp/steer16.jsonl')]
print('changed outputs:', sum(x['output']!=y['output'] for x,y in zip(b,s)), 'of', len(b))
for x,y in list(zip(b,s))[:2]: print('\n[',x['language'],x['label'],']', x['prompt'][:80], '\n  base :', x['output'][:160].replace('\n',' '), '\n  steer:', y['output'][:160].replace('\n',' '))
EOF
```

Expected: VRAM under 16 GB, and at least some outputs differ between base and steered.

**Cell 6 — the real runs (day 2).** Push after each so nothing is lost if the session dies.

```python
!python -m ayasteer.gpu direction --variant en    --out results/directions/en
!python -m ayasteer.gpu direction --variant multi --out results/directions/multi
!python -m ayasteer.gpu push --repo uzainahmed/tiny-aya-mlsteer-runs --path results
!python -m ayasteer.gpu generate --split eval --condition base --out results/gen/eval/base.jsonl
!python -m ayasteer.gpu push --repo uzainahmed/tiny-aya-mlsteer-runs --path results
!python -m ayasteer.gpu sweep --split dev --variants en,multi --strengths 0.5,1,2,4
!python -m ayasteer.gpu push --repo uzainahmed/tiny-aya-mlsteer-runs --path results
```

Then stop: the judge runs on the laptop and picks the strengths. Cell 7 is filled in after that.

**Cell 7 — final conditions (day 3, strengths from `results/report/selected_strengths.json`).**

```python
S_EN, S_MULTI = 2.0, 2.0   # replace with the selected values
!python -m ayasteer.gpu generate --split eval --condition steer_en    --strength {S_EN}    --direction results/directions/en    --out results/gen/eval/steer_en.jsonl
!python -m ayasteer.gpu generate --split eval --condition steer_multi --strength {S_MULTI} --direction results/directions/multi --out results/gen/eval/steer_multi.jsonl
!python -m ayasteer.gpu generate --split eval --condition ablate      --strength 1.0       --direction results/directions/en    --out results/gen/eval/ablate.jsonl
!python -m ayasteer.gpu capability --conditions base,steer_en:{S_EN},steer_multi:{S_MULTI},ablate:1.0
!python -m ayasteer.gpu redteam    --conditions base,steer_en:{S_EN},steer_multi:{S_MULTI},ablate:1.0
!python -m ayasteer.gpu push --repo uzainahmed/tiny-aya-mlsteer-runs --path results
```
