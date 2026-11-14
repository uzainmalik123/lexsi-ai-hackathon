"""Static figures for the article. Palette and mark rules follow the dataviz reference palette:
categorical hues in fixed order, thin marks, recessive axes, text in ink colors, one axis per chart."""
from __future__ import annotations

from pathlib import Path

from ayasteer.schema import HEADLINE, LANG_NAME

SERIES = {"base": "#2a78d6", "steer_en": "#eb6834", "steer_multi": "#1baf7a", "ablate": "#898781",
          "ablate_en": "#eda100", "ablate_multi": "#8f5bd6"}
LABEL = {"base": "base", "steer_en": "steer (en direction)", "steer_multi": "steer (multi direction)", "ablate": "ablated (layer 24)",
         "ablate_en": "ablated (en direction)", "ablate_multi": "ablated (multi direction)"}
SURFACE, INK, INK2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _style(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w=8.5, h=4.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), dpi=160, facecolor=SURFACE)
    return plt, fig, ax


def sweep_figure(sweep: dict[tuple[str, float], dict], base: dict, out: Path) -> Path:
    """Three curves (harmful compliance, over-refusal, benign incoherent) per direction, vs strength."""
    import matplotlib.pyplot as plt  # noqa: F401
    plt, fig, axes = None, None, None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants = sorted({v for v, _ in sweep})
    fig, axes = plt.subplots(1, len(variants), figsize=(4.6 * len(variants), 3.8), dpi=160, facecolor=SURFACE,
                             sharey=True, squeeze=False)
    metrics = [("harmful_refusal", SERIES["base"], "refusal on harmful"),
               ("harmful_compliance", SERIES["ablate_en"], "harmful compliance"),
               ("over_refusal", SERIES["steer_en"], "over-refusal (benign)"),
               ("benign_incoherent", SERIES["steer_multi"], "incoherent (benign)")]
    for ax, v in zip(axes[0], variants):
        xs = sorted(s for vv, s in sweep if vv == v)
        for key, color, label in metrics:
            ys = [sweep[(v, s)]["__mean__"][key] for s in xs]
            ax.plot([0.0] + xs, [base["__mean__"][key]] + ys, color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        _style(ax, f"{v} direction", "rate")
        ax.set_xlabel("steering strength (0 = base)", color=INK2, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Strength sweep on the dev split (mean over headline languages)", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE); plt.close(fig)
    return out


def before_after_figure(results: dict[str, dict[str, dict]], metric: str, out: Path, title: str) -> Path:
    plt, fig, ax = _fig(9.5, 4.2)
    conds = [c for c in SERIES if c in results]
    langs = [l for l in HEADLINE if l in results["base"]]
    width = 0.8 / len(conds)
    for i, c in enumerate(conds):
        xs = [j + (i - (len(conds) - 1) / 2) * width for j in range(len(langs))]
        ys = [results[c].get(l, {}).get(metric, float("nan")) for l in langs]
        ax.bar(xs, ys, width=width * 0.92, color=SERIES[c], label=LABEL[c], linewidth=0)
    ax.set_xticks(range(len(langs))); ax.set_xticklabels([LANG_NAME[l] for l in langs])
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    _style(ax, title + "\n", metric.replace("_", " "))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=len(conds), loc="lower left",
              bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE); plt.close(fig)
    return out


def gap_figure(base: dict[str, dict], out: Path) -> Path:
    plt, fig, ax = _fig(8.5, 3.6)
    langs = [l for l in HEADLINE if l in base and l != "en"]
    en = base["en"]["harmful_compliance"]
    gaps = [base[l]["harmful_compliance"] - en for l in langs]
    ax.bar(range(len(langs)), gaps, width=0.6, color=[SERIES["steer_en"] if g > 0 else SERIES["steer_multi"] for g in gaps], linewidth=0)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(range(len(langs))); ax.set_xticklabels([LANG_NAME[l] for l in langs])
    for i, g in enumerate(gaps):
        ax.text(i, g + (0.01 if g >= 0 else -0.01), f"{100 * g:+.0f}", ha="center",
                va="bottom" if g >= 0 else "top", fontsize=8, color=INK2)
    _style(ax, f"Base model: harmful compliance gap vs English (English = {100 * en:.0f}%)", "gap (points)")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100 * y:+.0f}"))
    fig.tight_layout()
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE); plt.close(fig)
    return out
