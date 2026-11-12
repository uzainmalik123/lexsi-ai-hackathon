# Dataset Card: ayasteer_benchmark

## Summary

- **Run timestamp**: 2026-11-12T09:48:07.877749Z
- **Wall clock time**: 4.0s
- **Pipeline config hash**: `22acc82be84515eb`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 2476 | 2476 | — | 0 | 2476 |
| SchemaGate | 2476 | 2476 | 0 | 0 | 2476 |
| ExactDeduplicator | 2476 | 2476 | — | — | 2476 |
| MinHashDeduplicator | 2476 | 2476 | — | — | 2476 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 2476/0 passed (N/A), 0 rejected
- **SchemaGate**: 2476/2476 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - near_duplicates_removed: 0
  - surviving_count: 2476

## Known Limitations

- MinHash threshold used: **0.85** — this is a starting point, not a universal answer. Adjust per domain (legal, medical, and other domain-specific corpora may need a lower threshold to avoid losing structurally similar but semantically valid examples).
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.27
- **python**: 3.12.14
