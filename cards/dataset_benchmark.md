---
license: cc-by-nc-4.0
language: [en, hi, ar, zh, es, fr, ru, ja, th, bn, sw, vi, ko, ur]
task_categories: [text-generation]
tags: [safety, red-teaming, multilingual, refusal, lexsi-hackathon, curatorkit, cohere, tiny-aya]
pretty_name: Tiny Aya Multilingual Safety Bench
---

# Tiny Aya Multilingual Safety Bench

**Content warning.** This dataset contains harmful requests (weapons, cybercrime, self-harm, hate, fraud and more)
in 14 languages, because it is a safety benchmark. It is for evaluating and improving refusal behaviour.

A parallel multilingual safety benchmark built for the Lexsi AI Safety and Alignment Hackathon (Nov 2026) to measure
whether Cohere's Tiny Aya Global refuses harmful requests equally across languages, and whether it over-refuses
borderline benign ones. Curated with [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) (exact + MinHash dedup,
provenance manifest, checksums). Row counts: {{counts}}.

## Files

| file | what |
|---|---|
| `eval.jsonl` | the evaluation set: PolyGuard harmful + benign (same ids in every headline language), XSTest borderline-benign (English original + machine translations), MultiJail harmful (bn, sw, vi, ko), Urdu slices |
| `dev.jsonl` | small parallel PolyGuard set used only to choose the steering strength |
| `contrast_en.jsonl`, `contrast_multi.jsonl` | 64 harmful + 64 benign prompts (English / mixed-language) used to extract the refusal direction; never evaluated |
| `translations/*.jsonl` | the machine-translated rows (`translated_by: ollama/qwen3.5-9b`) |
| `back_translation_sample.json` | 10% of the translations round-tripped back to English for a sanity check |
| `manifest.json`, `lexsi_provenance.json`, `checksums.txt`, `dataset_card.md` | CuratorKIT provenance |

Schema: `id, source, language, label (harmful|benign), prompt, en_prompt, category, split, translated_by`.

## Sources and licences

- [ToxicityPrompts/PolyGuardPrompts](https://huggingface.co/datasets/ToxicityPrompts/PolyGuardPrompts) (test split, non-adversarial rows) — parallel prompts in 17 languages with harm labels.
- [DAMO-NLP-SG/MultiJail](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail) — human-translated harmful prompts.
- [Paul/XSTest](https://huggingface.co/datasets/Paul/XSTest) — the 250 safe prompts that look dangerous.

Please respect each source's licence; this card is released CC BY-NC 4.0 to match the most restrictive source.

## The judge rubric (frozen)

System prompt:

```
{{rubric_system}}
```

User prompt template:

```
{{rubric}}
```

## Limitations

- XSTest and Urdu rows are machine-translated by a local 9B model; only a 10% back-translation sample and a
  20-row human check in Urdu were done.
- Harm labels come from the source datasets; we did not re-annotate.
- MultiJail languages have no benign counterpart, so over-refusal is not measured there.
