from ayasteer.report.roi import roi


def test_roi_self_hosted_cost_scales_with_tokens():
    a = roi(tokens_per_s=100.0, gpu_usd_per_hour=0.5, requests=1_000_000, in_tokens=200, out_tokens=100,
            closed_in_usd_per_m=3.0, closed_out_usd_per_m=15.0)
    b = roi(tokens_per_s=100.0, gpu_usd_per_hour=0.5, requests=2_000_000, in_tokens=200, out_tokens=100,
            closed_in_usd_per_m=3.0, closed_out_usd_per_m=15.0)
    assert b["self_hosted_usd"] == 2 * a["self_hosted_usd"]
    assert b["closed_usd"] == 2 * a["closed_usd"]
    # 1M requests x 100 output tokens = 1e8 tokens / 100 tok/s = 1e6 s = 277.8 h x $0.5
    assert round(a["self_hosted_usd"], 2) == round(1e8 / 100 / 3600 * 0.5, 2)
    assert a["closed_usd"] == 1_000_000 * (200 * 3.0 + 100 * 15.0) / 1_000_000
    assert a["savings_ratio"] == a["closed_usd"] / a["self_hosted_usd"]
