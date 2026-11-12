---
license: cc-by-nc-4.0
language: [en, hi, ar, zh, es, fr, ru, ja, th, bn, sw, vi, ko, ur]
tags: [safety, red-teaming, multilingual, refusal, activation-steering, lexsi-hackathon, safetune, auditkit, cohere, tiny-aya]
pretty_name: Tiny Aya Multilingual Steering Runs
---

# Tiny Aya Multilingual Steering: runs, judge outputs, reports

**Content warning.** Contains Tiny Aya Global's raw outputs to harmful requests in 14 languages, including the
"ablated" condition where the refusal direction was removed on purpose to show it is causal. Nothing is redacted,
because the judge labels have to be checkable against the text.

Everything produced by the pipeline in
[github.com/uzainmalik123/lexsi-ai-hackathon](https://github.com/uzainmalik123/lexsi-ai-hackathon):

| folder | what |
|---|---|
| `gen/eval/*.jsonl`, `gen/dev/*.jsonl` | generations per condition (`base`, `steer_en`, `steer_multi`, `ablate`) and the dev-split strength sweep; `.meta.json` carries tokens/s on the GPU used |
| `judged/**` | the same rows with `judge_label` (refusal / compliance / partial / incoherent / unknown), the raw judge reply, and the two cheap detectors; `*.auditkit.json` is the AuditKIT RunResult fingerprint per file |
| `judge_agreement*.json/jsonl` | with vs without the English parallel prompt, per language |
| `spotcheck.jsonl` | the human-checked rows (English and Urdu) |
| `capability/*.json` | Belebele accuracy per language and condition (AuditKIT `evaluate`) |
| `redteam/*.json` | AuditKIT built-in red-team probes per condition |
| `directions/*` | the steering directions (also published as model repos) |
| `report/` | tables, figures, `selected_strengths.json`, `results.json`, ROI |

Judge: local Ollama `qwen3.5-9b` (Q4_K_M), temperature 0, thinking off, verdict-only reply. Rubric:

```
{{rubric_system}}
```

```
{{rubric}}
```
