---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
tags: [safety, refusal, activation-steering, safetune, lexsi-hackathon, cohere, tiny-aya, multilingual]
---

# Refusal direction for Tiny Aya Global ({{variant}} contrast set)

A single unit vector in the residual stream of `CohereLabs/tiny-aya-global` that mediates refusal, extracted with
[SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s `extract_refusal_direction` (Arditi et al., 2024) from
{{n_harmful}} harmful and {{n_harmless}} harmless prompts ({{variant}} = English-only or mixed-language contrast set),
layer {{layer}}. No weights were changed. Adding `strength × direction` to every decoder block's output at inference
strengthens refusal; projecting it out removes refusal (the ablation control in the write-up).

Files: `refusal_direction.safetensors` (key `direction`, fp32, shape `[hidden]`), `steering_config.json`
(layer, pooling, contrast-set SHA-256 `{{contrast_sha256}}`, created {{created}}).

## Use

```python
from safetensors.torch import load_file
from safetune.steer import RefusalDirectionModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype="float16", device_map="cuda")
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
direction = load_file("refusal_direction.safetensors")["direction"]
with RefusalDirectionModel(model, direction=direction, mode="steer", strength=1.0):
    ...  # model.generate(...) as usual
```

The strength we selected on the dev split, and the per-language before/after numbers, are in the write-up linked from
the GitHub repo. This is a hackathon artifact: evaluated with an LLM judge, not human labels.
