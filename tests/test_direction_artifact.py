import json
from pathlib import Path

from ayasteer.gpu.direction_meta import contrast_sha256, load_meta, save_meta


def test_steering_config_roundtrip(tmp_path: Path):
    meta = {"layer": 24, "n_harmful": 64, "n_harmless": 64, "pool_method": "last_token", "sweep_used": False}
    rows = [{"id": "a", "prompt": "x", "label": "harmful"}, {"id": "b", "prompt": "y", "label": "benign"}]
    save_meta(meta, tmp_path, model_id="CohereLabs/tiny-aya-global", contrast_rows=rows, variant="en")
    cfg = json.loads((tmp_path / "steering_config.json").read_text())
    assert cfg["layer"] == 24 and cfg["model_id"] == "CohereLabs/tiny-aya-global" and cfg["variant"] == "en"
    assert cfg["contrast_sha256"] == contrast_sha256(rows) and len(cfg["contrast_sha256"]) == 64
    assert cfg["created"]
    assert load_meta(tmp_path) == cfg


def test_contrast_sha_is_order_and_content_sensitive():
    a = [{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}]
    assert contrast_sha256(a) != contrast_sha256(list(reversed(a)))
    assert contrast_sha256(a) != contrast_sha256([{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "z"}])
