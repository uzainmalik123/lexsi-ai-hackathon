"""Self-hosted Tiny Aya + steering vs a closed API, for a realistic monthly volume.

Comparator list prices are CLI inputs with defaults recorded here; re-check the vendor
pricing page on the day of the write-up and record the date in the report.
"""
from __future__ import annotations

# Defaults (USD). GPU: cloud on-demand T4 (the Kaggle class of card). Closed model: a
# mid-tier frontier API's list price per million tokens. Both are overridable from the CLI.
DEFAULT_GPU_USD_PER_HOUR = 0.35
DEFAULT_CLOSED_IN_USD_PER_M = 3.0
DEFAULT_CLOSED_OUT_USD_PER_M = 15.0


def roi(tokens_per_s: float, gpu_usd_per_hour: float, requests: int, in_tokens: int, out_tokens: int,
        closed_in_usd_per_m: float, closed_out_usd_per_m: float) -> dict:
    gen_tokens = requests * out_tokens
    gpu_hours = gen_tokens / tokens_per_s / 3600
    self_hosted = gpu_hours * gpu_usd_per_hour
    closed = requests * (in_tokens * closed_in_usd_per_m + out_tokens * closed_out_usd_per_m) / 1_000_000
    return {
        "requests": requests, "in_tokens": in_tokens, "out_tokens": out_tokens,
        "measured_tokens_per_s": tokens_per_s, "gpu_usd_per_hour": gpu_usd_per_hour,
        "gpu_hours": round(gpu_hours, 2), "self_hosted_usd": self_hosted, "closed_usd": closed,
        "savings_ratio": closed / self_hosted if self_hosted else float("inf"),
        "note": "self-hosted cost counts generated tokens only (prompt processing is a small fraction on a 3B model); "
                "steering adds one hook per layer and no measurable throughput cost",
    }
