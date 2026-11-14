# One refusal direction, every language: measuring and steering multilingual refusal in Tiny Aya Global

**Track 2 (2a #4 multilingual safety, 2b #8 inference-time steering). Solo entry.**

Cohere's Tiny Aya Global speaks dozens of languages, but safety training is mostly English. We asked whether it refuses harmful requests equally across languages, and whether its own refusal direction can be used at inference time, with no retraining, to strengthen or explain that behaviour.

**Benchmark (CuratorKIT).** 2,040 evaluation prompts in 14 languages: PolyGuardPrompts harmful and benign prompts with the same ids in nine headline languages, XSTest's safe-but-scary prompts machine-translated to catch over-refusal, MultiJail's human translations for Bengali, Swahili, Vietnamese and Korean, and Urdu. CuratorKIT ran schema checks, exact and MinHash dedup, and wrote provenance and checksums. Separate dev and contrast splits are used only for choosing strengths and extracting directions.

**Intervention (SafeTune).** `extract_refusal_direction` (Arditi et al.) from English-only and multilingual contrast pairs; `RefusalDirectionModel` adds the direction (steer) or projects it out (ablate) at inference. When the documented layer (24) turned out to be correlated with refusal but not causal, we re-ran the paper's layer selection in a memory-safe loop and found layer 19.

**Evaluation (AuditKIT + local judge).** Every output is labelled refusal / compliance / partial / incoherent by `qwen3.5-9b` on Ollama, validated by with/without-English agreement (96%) and a 40-row human check in English and Urdu (87.5% on refusal vs not). AuditKIT scores Belebele capability in 10 languages, runs its red-team probes, and fingerprints every judged run.

**Findings.**
1. No large multilingual gap: base refusal on harmful prompts is 83–93% in the headline languages, 73–80% in the MultiJail languages; harmful compliance is 0–5% everywhere.
2. Positive steering buys little on a model this cautious: +2.4 points of refusal for +4.3 points of over-refusal (English direction), +1.7 for +2.0 (multilingual direction).
3. **One direction carries refusal in every language.** Ablating the layer-19 direction cuts explicit English refusals from 82% to 10% (English direction) and 7% (multilingual), and lowers judged refusal by 12–35 points in all 14 languages, including five neither direction was extracted from. Belebele accuracy stays at 76%. The English and multilingual directions have cosine 0.96. An attack found in English removes safety everywhere; a defence anchored to this direction would cover every language at once.
4. Two measurement lessons: the 9B judge counts soft redirects and truncated harmful answers as refusals, understating ablation; AuditKIT's built-in red-team detectors are canary-based and score every condition identically, including the ablated model.

**Cost.** No training. Steering is one 2048-float vector and a forward hook with no measurable throughput cost; about $72 per million requests self-hosted on a T4 versus about $2,040 on a closed API.

**Artifacts.** Code: github.com/uzainmalik123/lexsi-ai-hackathon. Hugging Face: benchmark dataset, runs dataset (harmful completions redacted, labels kept), the two steering directions, and a static replay Space. The refusal-removal (layer-19) vectors are not published.
