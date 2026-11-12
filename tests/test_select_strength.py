import pytest

from ayasteer.report.select_strength import select_strength


def _rates(hc, orr, inc):
    return {"__mean__": {"harmful_compliance": hc, "over_refusal": orr, "benign_incoherent": inc}}


BASE = _rates(0.40, 0.05, 0.02)


def test_select_strength_picks_lowest_compliance_among_valid():
    sweep = {
        ("en", 0.5): _rates(0.30, 0.06, 0.02),
        ("en", 1.0): _rates(0.20, 0.10, 0.03),
        ("en", 2.0): _rates(0.10, 0.14, 0.05),
        ("en", 4.0): _rates(0.02, 0.30, 0.04),  # over-refusal budget blown
    }
    assert select_strength(sweep, BASE) == {"en": 2.0}


def test_select_strength_rejects_over_budget():
    sweep = {("en", 1.0): _rates(0.05, 0.20, 0.0)}  # +15 points over base 0.05
    with pytest.raises(ValueError):
        select_strength(sweep, BASE)


def test_select_strength_rejects_incoherent():
    sweep = {("en", 1.0): _rates(0.01, 0.05, 0.30), ("en", 0.5): _rates(0.25, 0.05, 0.05)}
    assert select_strength(sweep, BASE) == {"en": 0.5}


def test_select_strength_per_variant():
    sweep = {("en", 1.0): _rates(0.2, 0.08, 0.0), ("multi", 1.0): _rates(0.3, 0.08, 0.0),
             ("multi", 2.0): _rates(0.1, 0.12, 0.01)}
    assert select_strength(sweep, BASE) == {"en": 1.0, "multi": 2.0}
