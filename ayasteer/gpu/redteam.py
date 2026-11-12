"""AuditKIT's built-in red-team probes (injection, jailbreak, encoding, over-refusal) per condition."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def run_redteam(model, tok, condition: str, strength: float, direction, out_path: Path) -> dict:
    from auditkit.redteam import RedTeamRunner
    from ayasteer.gpu.generate import MODE_FOR_CONDITION
    from ayasteer.gpu.model import generate_batch, steering_context

    def callable_model(prompts: list[str]) -> list[str]:
        outs = []
        for i in range(0, len(prompts), 16):
            outs.extend(t for t, _ in generate_batch(model, tok, prompts[i:i + 16], max_new_tokens=96))
        return outs

    with steering_context(model, direction, MODE_FOR_CONDITION[condition], strength):
        result = RedTeamRunner(callable_model).run()

    per_probe: dict[str, dict] = {}
    for r in result.results:
        p = per_probe.setdefault(r.probe_name, {"total": 0, "passed": 0, "invalid": 0})
        p["total"] += 1
        p["passed"] += int(bool(r.passed))
        p["invalid"] += int(bool(getattr(r, "invalid", False)))
    out = {
        "condition": condition, "strength": strength,
        "pass_rate": result.pass_rate, "total": result.total_probes,
        "per_probe": per_probe,
        "rows": [{"probe": r.probe_name, "detector": r.detector_name, "prompt": r.prompt,
                  "output": r.output, "passed": r.passed} for r in result.results],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log.info("redteam %s s=%g pass_rate=%s", condition, strength, result.pass_rate)
    return {k: v for k, v in out.items() if k != "rows"}
