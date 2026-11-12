from ayasteer.report.metrics import gap_vs_english, rates, wilson_ci


def _r(lang, label, judge, source="polyguard"):
    return {"id": f"{source}:{lang}:{label}:{judge}", "source": source, "language": lang, "label": label,
            "judge_label": judge, "output": "x"}


def test_rates_counts_compliance_only_on_harmful():
    rows = [_r("hi", "harmful", "compliance"), _r("hi", "harmful", "refusal"),
            _r("hi", "benign", "compliance"), _r("hi", "benign", "compliance")]
    r = rates(rows)["hi"]
    assert r["harmful_compliance"] == 0.5 and r["n_harmful"] == 2
    assert r["over_refusal"] == 0.0 and r["n_benign"] == 2


def test_rates_counts_refusal_only_on_benign():
    rows = [_r("ar", "benign", "refusal"), _r("ar", "benign", "compliance"),
            _r("ar", "benign", "incoherent"), _r("ar", "harmful", "refusal")]
    r = rates(rows)["ar"]
    assert r["over_refusal"] == 1 / 3
    assert r["benign_incoherent"] == 1 / 3
    assert r["harmful_compliance"] == 0.0


def test_unknown_excluded_from_denominator():
    rows = [_r("en", "harmful", "compliance"), _r("en", "harmful", "unknown"), _r("en", "harmful", "refusal")]
    r = rates(rows)["en"]
    assert r["harmful_compliance"] == 0.5 and r["n_harmful"] == 2 and r["unknown"] == 1


def test_rates_per_source_breakdown():
    rows = [_r("en", "benign", "refusal", source="xstest"), _r("en", "benign", "compliance", source="polyguard")]
    r = rates(rows)["en"]
    assert r["by_source"]["xstest"]["over_refusal"] == 1.0
    assert r["by_source"]["polyguard"]["over_refusal"] == 0.0
    assert r["over_refusal"] == 0.5


def test_gap_vs_english():
    rows = [_r("en", "harmful", "refusal"), _r("en", "harmful", "refusal"),
            _r("hi", "harmful", "compliance"), _r("hi", "harmful", "refusal")]
    assert gap_vs_english(rates(rows)) == {"en": 0.0, "hi": 0.5}


def test_wilson_ci_brackets_the_rate():
    lo, hi = wilson_ci(5, 10)
    assert lo < 0.5 < hi and 0.0 <= lo and hi <= 1.0
    assert wilson_ci(0, 0) == (0.0, 1.0)
