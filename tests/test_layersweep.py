from ayasteer.gpu.layersweep import is_refusal, pick_layer, split_validation


def test_is_refusal_matches_arditi_substrings():
    assert is_refusal("I can’t help with that request.")
    assert is_refusal("  I'm sorry, but I won't do that")
    assert not is_refusal("Here is a recipe for banana bread.")


def test_pick_layer_lowest_bypass_under_kl():
    table = [{"layer": 10, "bypass": 0.05, "kl": 0.5},   # over kl_threshold
             {"layer": 12, "bypass": 0.20, "kl": 0.05},
             {"layer": 14, "bypass": 0.30, "kl": 0.02}]
    assert pick_layer(table, baseline=0.9)["layer"] == 12


def test_pick_layer_falls_back_to_looser_kl():
    table = [{"layer": 10, "bypass": 0.05, "kl": 0.5}, {"layer": 12, "bypass": 0.80, "kl": 0.05}]
    assert pick_layer(table, baseline=0.9)["layer"] == 10


def test_pick_layer_none_when_nothing_halves_refusal():
    table = [{"layer": 12, "bypass": 0.70, "kl": 0.05}]
    assert pick_layer(table, baseline=0.9) is None


def _row(i, label, en):
    return {"id": i, "label": label, "en_prompt": en}


def test_split_validation_holds_out_contrast_head_and_dev():
    contrast = [_row(f"h{i}", "harmful", f"H{i}") for i in range(4)] + [_row(f"b{i}", "benign", f"B{i}") for i in range(4)]
    dev = [_row("d1", "harmful", "DH"), _row("d1", "harmful", "DH"), _row("d2", "benign", "DB"), _row("d3", "harmful", "H0")]
    vh, vb, held = split_validation(contrast, dev, n_val=2)
    assert vh == ["H0", "H1", "DH"] and vb == ["B0", "B1", "DB"]
    assert held == {"H0", "H1", "DH", "B0", "B1", "DB"}
