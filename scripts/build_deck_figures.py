"""Build deck-styled, inline-ready SVG figures for the Part I presentation.

Real data-science figures (labeled axes, scales, bootstrap CIs) regenerated from
the run data in ``outputs/runs/`` via the same per-record frames the notebooks
use. Output: self-contained SVGs in ``docs/presentation/assets/`` that the deck
builder inlines (so the deck's IBM Plex web fonts render the labels).

Run from the data-bearing checkout (outputs/ + data/cache live there)::

    DECK_RUNS=/abs/path/to/outputs/runs \
      uv run --directory /abs/path/to/repo python scripts/build_deck_figures.py

``DECK_RUNS`` defaults to ``<repo>/outputs/runs``. Figures write to
``<this-repo>/docs/presentation/assets``. Each figure is isolated: one failing
figure logs and is skipped, the rest still build.
"""
from __future__ import annotations
import json, os, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scrye.scoring import bootstrap_ci

# --- paths -------------------------------------------------------------------
ASSETS = Path(__file__).resolve().parents[1] / "docs" / "presentation" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
RUNS = Path(os.environ.get("DECK_RUNS", Path(__file__).resolve().parents[1] / "outputs" / "runs"))

# --- palette (matches the deck design system) --------------------------------
INK = "#1B1A17"; SEC = "#8C8579"; GREEN = "#2E8568"; GREEN_L = "#5FCBA0"
AMBER = "#B5821E"; GREY = "#A39D92"; PAPERLINE = "#DAD4C8"
LIGHTINK = "#ECE7DC"; DARKSEC = "#97928A"

plt.rcParams.update({
    "svg.fonttype": "none",                       # keep text as <text> -> web fonts
    "font.family": ["IBM Plex Sans", "DejaVu Sans"],
    "font.size": 14,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.1, "axes.titlesize": 16, "axes.titleweight": "medium",
    "axes.labelsize": 14, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "legend.frameon": False, "legend.fontsize": 12.5,
})


def C(dark: bool) -> dict:
    """Color set for a light or dark slide."""
    if dark:
        return dict(ink=LIGHTINK, sec=DARKSEC, green=GREEN_L, grey="#6E6A62",
                    truth=DARKSEC, pred=GREEN_L, amber="#D6A43E", line="#3A3833")
    return dict(ink=INK, sec=SEC, green=GREEN, grey=GREY,
                truth=GREY, pred=GREEN, amber=AMBER, line=PAPERLINE)


def _apply(ax, c):
    ax.tick_params(colors=c["ink"]); ax.yaxis.label.set_color(c["ink"])
    ax.xaxis.label.set_color(c["ink"]); ax.title.set_color(c["ink"])
    for s in ax.spines.values():
        s.set_color(c["sec"])


def save(fig, name):
    p = ASSETS / f"{name}.svg"
    fig.savefig(p, bbox_inches="tight", transparent=True, dpi=150)
    plt.close(fig)
    # matplotlib can't resolve IBM Plex locally; force it via a <style> that the
    # deck's web fonts satisfy once the SVG is inlined into the page.
    style = ("<style>text,tspan{font-family:'IBM Plex Sans','IBM Plex Mono',"
             "sans-serif !important;}</style>")
    txt = p.read_text().replace("</svg>", style + "</svg>", 1)
    p.write_text(txt)
    print(f"  ✓ {name}.svg ({p.stat().st_size} B)")


def load(name):
    return json.load(open(RUNS / f"{name}.results.json"))


def frame(records):
    df = pd.DataFrame(records)
    if "is_population" in df:
        df["splitname"] = np.where(df["is_population"], "pop", "grouped")
    return df


def mean_ci(scores):
    return bootstrap_ci(list(scores))


# =============================================================================
# Figures
# =============================================================================

def fig_score_hist():
    df = frame(load("2026-06-21-decomp-final")["faithful"])
    c = C(False)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    s = df["score"].clip(lower=-60)
    ax.hist(s, bins=36, color=GREEN, alpha=0.85, edgecolor="white", linewidth=0.4)
    m = df["score"].mean()
    ax.axvline(0, color=GREY, lw=1.6, ls="--")
    ax.axvline(m, color=INK, lw=1.8)
    ax.text(0, ax.get_ylim()[1]*0.96, " uniform (0)", color=SEC, fontsize=12, va="top")
    ax.text(m, ax.get_ylim()[1]*0.96, f" mean {m:.1f}", color=INK, fontsize=12.5, va="top", fontweight="medium")
    ax.set_xlabel("SimBench score  S   (0 = uniform · 100 = perfect)")
    ax.set_ylabel("questions")
    _apply(ax, c)
    save(fig, "07_score_hist")


def fig_score_vs_entropy():
    df = frame(load("2026-06-21-decomp-final")["faithful"])
    c = C(False)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.scatter(df["truth_entropy"], df["score"].clip(lower=-60), s=12,
               color=GREEN, alpha=0.28, edgecolors="none", rasterized=True)
    # binned mean
    bins = np.linspace(0, 1, 9)
    idx = np.digitize(df["truth_entropy"], bins) - 1
    xs, ys = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= 5:
            xs.append((bins[b] + bins[b+1]) / 2); ys.append(df["score"][m].mean())
    ax.plot(xs, ys, color=INK, lw=2.4, marker="o", ms=6, label="binned mean")
    ax.axhline(0, color=GREY, lw=1.4, ls="--")
    ax.set_xlabel("truth entropy   (0 = consensus → 1 = divided)")
    ax.set_ylabel("SimBench score  S")
    ax.legend(loc="upper right")
    _apply(ax, c)
    save(fig, "07_score_vs_entropy")


def fig_examples():
    recs = load("2026-06-21-decomp-final")["faithful"]
    df = frame(recs)
    from scrye.decompose import decompose_error
    dec = pd.DataFrame([decompose_error(r["pred"], r["truth"], r.get("options")) for r in recs])
    df["concentration_err"] = dec["concentration_err"].values
    df["location_err"] = dec["location_err"].values
    d = df
    c = C(False)
    # pick three illustrative items
    def opt_dist(row, col):
        opts = list(row["options"])
        return opts, [float(row[col].get(o, 0.0)) for o in opts]
    picks = []
    if d["concentration_err"].notna().any():
        over = d.sort_values("concentration_err", ascending=False).iloc[0]
        loc = d.sort_values("location_err", ascending=False).iloc[0]
        bal = d.iloc[(d["score"] - d["score"].median()).abs().argsort()].iloc[0]
        picks = [("over-concentrated", over), ("wrong location", loc), ("typical", bal)]
    else:
        lo = d.sort_values("score").iloc[0]
        bal = d.iloc[(d["score"] - d["score"].median()).abs().argsort()].iloc[0]
        hi = d.sort_values("score").iloc[-1]
        picks = [("worst fit", lo), ("typical", bal), ("best fit", hi)]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
    for ax, (label, row) in zip(axes, picks):
        opts, P = opt_dist(row, "truth"); _, Q = opt_dist(row, "pred")
        x = np.arange(len(opts))
        ax.bar(x - 0.2, P, 0.4, color=GREY, label="human P")
        ax.bar(x + 0.2, Q, 0.4, color=GREEN, label="predicted Q")
        ax.set_title(f"{label}\nS = {row['score']:.0f}", fontsize=12.5)
        ax.set_xticks(x); ax.set_xticklabels([str(o)[:6] for o in opts], fontsize=9, rotation=20, ha="right")
        ax.set_ylim(0, max(max(P), max(Q)) * 1.25 + 1e-6)
        _apply(ax, c)
    axes[0].set_ylabel("probability"); axes[0].legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    save(fig, "03_examples")


def fig_calibration():
    df = frame(load("2026-06-21-decomp-final")["faithful"])
    c = C(False)
    pp, tp = [], []
    for _, row in df.iterrows():
        opts = list(row["options"])
        for o in opts:
            pp.append(float(row["pred"].get(o, 0.0))); tp.append(float(row["truth"].get(o, 0.0)))
    pp = np.array(pp); tp = np.array(tp)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(pp, tp, s=9, color=GREEN, alpha=0.15, edgecolors="none", rasterized=True)
    bins = np.linspace(0, 1, 11); idx = np.digitize(pp, bins) - 1
    bx, by = [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 10:
            bx.append((bins[b]+bins[b+1])/2); by.append(tp[m].mean())
    ax.plot(bx, by, color=INK, lw=2.4, marker="o", ms=6, label="binned mean")
    ax.plot([0, 1], [0, 1], color=GREY, lw=1.6, ls="--", label="perfect")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("empirical (human) probability")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left")
    _apply(ax, c)
    save(fig, "10_calibration")


def fig_error_decomp():
    data = load("2026-06-21-decomp-final")
    from scrye.decompose import decompose_error
    c = C(False)
    rows = []
    for sysname, label in [("faithful", "faithful\n@ 3.1"), ("final_router", "our\nsystem")]:
        dec = pd.DataFrame([decompose_error(r["pred"], r["truth"], r.get("options"))
                            for r in data[sysname]])
        rows.append((label, dec["concentration_err"].mean(), dec["location_err"].mean()))
    labels = [r[0] for r in rows]
    conc = [r[1] for r in rows]; loc = [r[2] for r in rows]
    totals = [a + b for a, b in zip(conc, loc)]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = np.arange(len(rows))
    ax.bar(x, conc, 0.5, color=GREEN, label="concentration  —  wrong spread (we stay too diffuse)")
    ax.bar(x, loc, 0.5, bottom=conc, color=AMBER, label="location  —  mass on the wrong options")
    for i, (cc, ll) in enumerate(zip(conc, loc)):
        ax.text(i, cc / 2, f"{cc:.2f}", ha="center", va="center", color="white", fontsize=12.5)
        ax.text(i, cc + ll / 2, f"{ll:.2f}", ha="center", va="center", color="white", fontsize=12.5)
        ax.text(i, totals[i] + 0.006, f"total TVD {totals[i]:.2f}", ha="center", fontsize=10.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, max(totals) * 1.16)
    ax.set_ylabel("mean error  (TVD = concentration + location)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=11)
    ax.set_title("Our gains are all on location — concentration (spread) is untouched", fontsize=12.5)
    _apply(ax, c)
    save(fig, "17_error_decomp")


def _grouped_mean_ci(df):
    g = df[df["splitname"] == "grouped"]["score"]
    return mean_ci(g)


def fig_graveyard_personas():
    data = load("2026-06-20-persona-mechanisms")
    c = C(False)
    want = [("faithful", "faithful (direct ask)", GREEN),
            ("anti_flattening", "anti-flattening", GREEN),
            ("contextualized", "contextualized", GREEN),
            ("diversity_elicitation", "diversity-elicitation CoT", GREY),
            ("monte_carlo", "Monte-Carlo individuals", GREY)]
    rows = []
    for key, label, col in want:
        if key in data:
            m, lo, hi = _grouped_mean_ci(frame(data[key]))
            rows.append((label, m, lo, hi, col))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    y = np.arange(len(rows))
    for i, (label, m, lo, hi, col) in enumerate(rows):
        ax.barh(i, m, color=col, alpha=0.92, height=0.62)
        ax.plot([lo, hi], [i, i], color=INK, lw=1.6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color=GREY, lw=1.3, ls="--")
    ax.set_xlabel("grouped SimBench score  S   (95% CI)")
    ax.text(0.98, 0.04, "1M-persona census electorate: 51.8 vs simple champion 63.2  (−11.4)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color=SEC, style="italic")
    _apply(ax, c)
    save(fig, "08_graveyard_personas")


def fig_voting():
    # authoritative point estimates from stage-13 (voting-sim summary)
    c = C(False)
    rows = [("Choices13k", 15.6, GREEN), ("OSPsychMACH", -26.0, GREY), ("MoralMachine", -157.9, GREY)]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], 0.55, color=[r[2] for r in rows])
    ax.axhline(0, color=INK, lw=1.3)
    ax.axhline(-2.73, color=AMBER, lw=1.6, ls="--", label="uniform baseline (−2.7)")
    for i, r in enumerate(rows):
        ax.text(i, r[1] + (4 if r[1] > 0 else -8), f"{r[1]:+.0f}", ha="center",
                va="bottom" if r[1] > 0 else "top", fontsize=12.5,
                color=GREEN if r[1] > 0 else INK)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("voting-sim Δ vs direct ask   (SimBench S)")
    ax.legend(loc="lower left")
    ax.set_title("Discrete voting: one transfer win, one catastrophe", fontsize=14)
    _apply(ax, c)
    save(fig, "09_voting")


def fig_model_sweep():
    data = load("2026-06-20-model-sweep")
    c = C(False)
    models = [("gemini-flash-lite", "gemini-2.5", GREY),
              ("gemini-3.1-flash-lite", "gemini-3.1", GREEN),
              ("gemini-3.5-flash", "gemini-3.5", AMBER)]
    strategies = ["faithful", "anti_flattening", "contextualized"]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    w = 0.26
    xb = np.arange(len(strategies))
    for j, (mkey, mlabel, col) in enumerate(models):
        means, los, his = [], [], []
        for s in strategies:
            k = f"{mkey}::{s}"
            m, lo, hi = _grouped_mean_ci(frame(data[k])) if k in data else (np.nan,)*3
            means.append(m); los.append(m-lo); his.append(hi-m)
        ax.bar(xb + (j-1)*w, means, w, color=col, label=mlabel,
               yerr=[los, his], capsize=3, error_kw=dict(lw=1.2, ecolor=INK))
    ax.set_xticks(xb); ax.set_xticklabels(["faithful", "anti-flattening", "contextualized"])
    ax.set_ylabel("grouped SimBench score  S   (95% CI)")
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("Fit, not size: gemini-3.1-flash-lite ≥ 3.5-flash on every strategy", fontsize=14)
    _apply(ax, c)
    save(fig, "14_model_sweep")


def fig_headline_decomp():
    fm = load("2026-06-21-TEST-faithful-models")
    tf = load("2026-06-21-TEST-final")["test"]
    lin = load("2026-06-21-TEST-lineage")
    c = C(False)
    # stages: Qwen faithful, faithful@3.1, our system (cc+abstain)
    def splitavg(g, p): return (g + p) / 2
    qwen = fm["faithful@Qwen2.5-72B"]
    stages = [
        ("faithful\n@ Qwen2.5-72B", qwen["grouped"], qwen["pop"], qwen["split_avg_S"], GREY),
        ("+ model\n@ gemini-3.1", tf["grouped"]["faithful"], tf["pop"]["faithful"],
         splitavg(tf["grouped"]["faithful"], tf["pop"]["faithful"]), GREEN_L),
        ("+ our method\n(cc + abstain)", lin["cc_abstain"]["grouped"], lin["cc_abstain"]["pop"],
         splitavg(lin["cc_abstain"]["grouped"], lin["cc_abstain"]["pop"]), GREEN),
    ]
    metrics = ["grouped", "pop", "split-avg"]
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    w = 0.26; xb = np.arange(len(metrics))
    shortlab = ["faithful @ Qwen2.5-72B", "+ gemini-3.1  (model)", "+ cc + abstain  (method)"]
    for j, (label, g, p, sa, col) in enumerate(stages):
        vals = [g, p, sa]
        ax.bar(xb + (j-1)*w, vals, w, color=col, label=shortlab[j])
        for k, v in enumerate(vals):
            ax.text(xb[k] + (j-1)*w, v + 0.7, f"{v:.0f}", ha="center", fontsize=10.5, color=INK)
    ax.set_xticks(xb); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 50)
    ax.set_ylabel("SimBench score  S")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=10.5)
    ax.set_title("Most of the lift is the model — and almost all of it on grouped", fontsize=13.5)
    _apply(ax, c)
    save(fig, "14_headline_decomp")


def fig_result():
    # split-avg progression. Points are the reported system (cc + abstain, from
    # TEST-lineage); CIs are combined from the tying final system's disjoint
    # pop/grouped strata (TEST-final): split-avg = ½(pop+grouped), and because the
    # strata are disjoint their bootstrap variances add → SE = ½·hypot(SE_g, SE_p).
    lin = load("2026-06-21-TEST-lineage")
    strat = load("2026-06-21-TEST-final")["test"]
    c = C(True)  # dark slide

    def se(ci):
        return (ci[1] - ci[0]) / (2 * 1.96)

    def sa(d):
        return (d["grouped"] + d["pop"]) / 2

    def sa_ci(point, gci, pci):
        h = 1.96 * 0.5 * np.hypot(se(gci), se(pci))
        return [point - h, point + h]

    fa = sa(lin["faithful"])         # faithful @ gemini-3.1 — split-avg ≈ 35.0
    ou = sa(lin["cc_abstain"])       # our system (cc + abstain) — split-avg ≈ 40.8
    fa_ci = sa_ci(fa, strat["grouped"]["faithful_ci"], strat["pop"]["faithful_ci"])
    ou_ci = sa_ci(ou, strat["grouped"]["final_ci"], strat["pop"]["final_ci"])
    delta = ou - fa
    dh = 1.96 * 0.5 * np.hypot(se(strat["grouped"]["delta_ci"]), se(strat["pop"]["delta_ci"]))
    d_ci = [delta - dh, delta + dh]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    # left: progression with CIs
    pts = [("faithful\n@ 3.1", fa, fa_ci, c["sec"]),
           ("our system", ou, ou_ci, c["green"])]
    x = np.arange(len(pts))
    for i, (lab, m, ci, col) in enumerate(pts):
        ax.bar(i, m, 0.5, color=col)
        ax.plot([i, i], ci, color=c["ink"], lw=1.8)
        ax.text(i, m + 1.0, f"{m:.1f}", ha="center", color=c["ink"], fontsize=13, fontweight="medium")
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pts])
    ax.set_ylabel("split-avg SimBench S")
    ax.set_title(f"Sealed test:  +{delta:.1f}   [{d_ci[0]:.1f}, {d_ci[1]:.1f}]",
                 fontsize=14, color=c["green"])
    _apply(ax, c)
    # right: paper reproduction number line
    ax2.set_xlim(20, 32); ax2.set_ylim(0, 1)
    ax2.hlines(0.5, 24.39, 29.35, color=c["green"], lw=7, alpha=0.55)
    ax2.plot(26.83, 0.5, "o", color=c["green"], ms=12)
    ax2.plot(27.61, 0.5, "|", color=c["ink"], ms=26, mew=3)
    ax2.text(26.83, 0.66, "ours 26.8", ha="center", color=c["green"], fontsize=12.5)
    ax2.text(27.61, 0.30, "paper 27.6", ha="center", color=c["ink"], fontsize=12.5)
    ax2.text(24.39, 0.36, "24.4", ha="center", color=c["sec"], fontsize=10)
    ax2.text(29.35, 0.36, "29.4", ha="center", color=c["sec"], fontsize=10)
    ax2.set_yticks([]); ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("split-avg S  @ Qwen2.5-72B")
    ax2.set_title("Paper reproduction ✓", fontsize=14, color=c["ink"])
    _apply(ax2, c)
    fig.tight_layout()
    save(fig, "13_result")


def fig_val_staircase():
    fvf = load("2026-06-21-faithful-vs-final")["overall"]
    c = C(False)
    # pooled val lineage (stage-17 val table); endpoints carry CIs from faithful-vs-final
    pts = [("faithful\n@ 3.1", 36.90, fvf["faithful_ci"]),
           ("anti-\nflattening", 43.43, None),
           ("cc +\nabstain", 45.38, None),
           ("router", 47.01, fvf["final_ci"])]
    gates = [None, "Stage 05 ✓", "Stage 12 ✓", "Stage 16 ✓"]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(pts)); y = [p[1] for p in pts]
    ax.step(x, y, where="mid", color=GREEN, lw=2.6)
    ax.scatter(x, y, color=GREEN, s=70, zorder=3)
    for i, (lab, v, ci) in enumerate(pts):
        ax.text(i, v + 0.7, f"{v:.1f}", ha="center", fontsize=12.5, color=INK, fontweight="medium")
        if ci:
            ax.plot([i, i], ci, color=INK, lw=1.6)
        if gates[i]:
            ax.text(i, 33.6, gates[i], ha="center", fontsize=10.5, color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pts])
    ax.set_ylabel("pooled validation S")
    ax.set_ylim(33, 49)
    ax.set_title("Each step cleared the η-gate before it counted", fontsize=14)
    _apply(ax, c)
    save(fig, "06_val_staircase")


def fig_cf_alignment():
    # Faithful (original) vs our conditioning, per stage — the requested comparison.
    # Per-system cf_alignment from the topline CSVs (baseline / fulldev-confirm / val-confirm).
    c = C(False)
    stages = ["Stage 01\n(dev)", "Stage 04\n(full dev)", "Stage 05\n(val)"]
    faithful = [0.217, 0.374, 0.397]
    ours = [0.217, 0.388, 0.419]          # anti_flattening — our conditioning
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(stages)); w = 0.38
    ax.bar(x - w/2, faithful, w, color=GREY, label="faithful (original)")
    ax.bar(x + w/2, ours, w, color=GREEN, label="our conditioning (anti-flattening)")
    for xi, (f, o) in enumerate(zip(faithful, ours)):
        ax.text(xi - w/2, f + 0.012, f"{f:.2f}", ha="center", fontsize=11, color=INK)
        ax.text(xi + w/2, o + 0.012, f"{o:.2f}", ha="center", fontsize=11, color=INK)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylabel("cf_alignment  (−1 wrong · 0 none · +1 right)")
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_title("Conditioning barely beats faithful (+0.00 to +0.02) — weak, and model-bound",
                 fontsize=12.5)
    _apply(ax, c)
    save(fig, "16_cf_alignment")


def fig_required_q():
    # Reconstruct the three required-question predicted distributions from cache
    # (cell 32 of notebook 03 — all LLM calls cached → deterministic, free).
    from scrye.data import load_all
    from scrye.splits import make_split
    import scrye.evaluate as ev
    from scrye.experiment import build_pipeline
    allr = load_all()
    full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    norms = ev.build_normalizers(full)
    req_recs = ev.required_question_records(split.test)
    final = build_pipeline(model="gemini-3.1-flash-lite", predictor="calibrated_commitment")
    req_results = {q: ev.evaluate(final, recs, normalizers=norms, progress=False)
                   for q, recs in req_recs.items()}
    c = C(False)
    nice = {"trust_president": "Trust in president", "gay_rights": "Gay rights",
            "internet_use": "Internet-use frequency"}

    def pooled(df, col, options, w):
        v = np.zeros(len(options))
        for wi, dist in zip(w, df[col]):
            for j, o in enumerate(options):
                v[j] += wi * float(dist.get(o, 0.0))
        s = v.sum()
        return v / s if s > 0 else v

    items = [(q, df) for q, df in req_results.items() if len(df)]
    fig, axes = plt.subplots(1, len(items), figsize=(4.9 * len(items), 4.0), squeeze=False)
    axes = axes[0]
    for ax, (q, df) in zip(axes, items):
        options = list(df.iloc[0]["options"])
        w = df["group_size"].to_numpy(float) if "group_size" in df else np.ones(len(df))
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(df))
        P = pooled(df, "truth", options, w); Q = pooled(df, "pred", options, w)
        x = np.arange(len(options))
        ax.bar(x - 0.2, P, 0.4, color=GREY, label="human P")
        ax.bar(x + 0.2, Q, 0.4, color=GREEN, label="predicted Q")
        ax.set_xticks(x); ax.set_xticklabels([str(o)[:11] for o in options], fontsize=9, rotation=22, ha="right")
        ax.set_ylim(0, max(P.max(), Q.max()) * 1.28 + 1e-6)
        ax.set_title(f"{nice.get(q, q)}\nS = {df['score'].mean():.0f}", fontsize=13)
        _apply(ax, c)
    axes[0].set_ylabel("probability"); axes[0].legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    save(fig, "15_required_q")
    for q, df in items:
        print(f"    [required-Q] {q}: S = {df['score'].mean():.1f}")


# Explicit SimBench dataset -> task-kind map (the heuristic classifier is the
# weak fallback; the deployed router used the LLM classifier). Aligned to the
# scrye.taskkind taxonomy.
DATASET_KIND = {
    "Afrobarometer": "opinion_survey", "ESS": "opinion_survey",
    "GlobalOpinionQA": "opinion_survey", "ISSP": "opinion_survey",
    "LatinoBarometro": "opinion_survey", "OpinionQA": "opinion_survey",
    "TISP": "opinion_survey", "DICES": "opinion_survey", "ConspiracyCorr": "opinion_survey",
    "NumberGame": "knowledge", "WisdomOfCrowds": "knowledge",
    "MoralMachine": "moral_dilemma", "MoralMachineClassic": "moral_dilemma",
    "Choices13k": "risky_choice",
    "OSPsychBig5": "personality_scale", "OSPsychMACH": "personality_scale",
    "OSPsychMGKT": "personality_scale", "OSPsychRWAS": "personality_scale",
    "ChaosNLI": "other", "Jester": "other",
}


def fig_cross_task():
    # Faithful baseline vs our system, absolute SimBench S, by task kind (legend).
    data = load("2026-06-21-decomp-final")
    c = C(False)
    order = ["opinion_survey", "personality_scale", "knowledge",
             "moral_dilemma", "risky_choice", "other"]
    nice = {"opinion_survey": "opinion survey", "personality_scale": "personality",
            "knowledge": "knowledge", "moral_dilemma": "moral dilemma",
            "risky_choice": "risky choice", "other": "other"}
    kcol = {"opinion_survey": GREEN, "personality_scale": "#6FA890",
            "knowledge": "#9DBBAF", "moral_dilemma": AMBER,
            "risky_choice": "#C99A3A", "other": GREY}
    systems = [("faithful", "faithful @ 3.1\n(baseline)"), ("final_router", "our system")]
    stats, ns = {}, {}
    for skey, _ in systems:
        df = frame(data[skey]); df["kind"] = df["dataset"].map(DATASET_KIND).fillna("other")
        for k in order:
            s = df[df["kind"] == k]["score"]
            stats[(skey, k)] = mean_ci(s) if len(s) else (np.nan, np.nan, np.nan)
            ns[k] = len(s)
    print("    [cross-task] mean S by kind (faithful -> ours):")
    for k in order:
        print(f"      {k:18s} {stats[('faithful', k)][0]:6.1f} -> {stats[('final_router', k)][0]:6.1f}  (n={ns[k]})")
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    nk = len(order); w = 0.82 / nk
    xb = np.arange(len(systems))
    for j, k in enumerate(order):
        means = [stats[(s, k)][0] for s, _ in systems]
        los = [stats[(s, k)][0] - stats[(s, k)][1] for s, _ in systems]
        his = [stats[(s, k)][2] - stats[(s, k)][0] for s, _ in systems]
        ax.bar(xb + (j - (nk - 1) / 2) * w, means, w, color=kcol[k], label=f"{nice[k]}  (n={ns[k]})",
               yerr=[los, his], capsize=2, error_kw=dict(lw=1.0, ecolor=INK))
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(xb); ax.set_xticklabels([s[1] for s in systems])
    ax.set_ylabel("SimBench score  S   (0 = uniform · 100 = perfect)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10)
    ax.set_title("By task kind — faithful baseline vs our system  (dev)", fontsize=13)
    _apply(ax, c)
    save(fig, "18_cross_task")


def fig_required_q_qwen():
    from scrye.data import load_all
    from scrye.splits import make_split
    import scrye.evaluate as ev
    from scrye.experiment import build_pipeline
    import scrye.llm as llm
    allr = load_all(); full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    norms = ev.build_normalizers(full)
    req_recs = ev.required_question_records(split.test)
    # cache-only guard: any cache miss raises instead of spending on the API.
    orig = llm.LLMClient.complete
    def cache_only(self, messages, **ov):
        payload = self._request_payload(messages, **ov); key = llm._cache_key(payload)
        hit = self._read_cache(key) if self.use_cache else None
        if hit is None:
            raise RuntimeError("CACHE_MISS — faithful@Qwen required-Q not cached")
        with self._lock:
            self.usage.cache_hits += 1
        return llm.LLMResponse(text=hit.get("text", ""), model=self.model, cached=True, raw=hit)
    llm.LLMClient.complete = cache_only
    try:
        qwen = build_pipeline(model="qwen-72b", predictor="simbench_faithful")
        ours = build_pipeline(model="gemini-3.1-flash-lite", predictor="calibrated_commitment")
        q_res = {q: ev.evaluate(qwen, recs, normalizers=norms, progress=False) for q, recs in req_recs.items()}
        o_res = {q: ev.evaluate(ours, recs, normalizers=norms, progress=False) for q, recs in req_recs.items()}
    finally:
        llm.LLMClient.complete = orig
    c = C(False)
    nice = {"trust_president": "Trust in president", "gay_rights": "Gay rights",
            "internet_use": "Internet-use frequency"}

    def pooled(df, col, options, w):
        v = np.zeros(len(options))
        for wi, dist in zip(w, df[col]):
            for j, o in enumerate(options):
                v[j] += wi * float(dist.get(o, 0.0))
        s = v.sum()
        return v / s if s > 0 else v

    items = [(q, q_res[q], o_res[q]) for q in q_res if len(q_res[q])]
    fig, axes = plt.subplots(1, len(items), figsize=(4.9 * len(items), 4.2), squeeze=False)
    axes = axes[0]
    for ax, (q, qd, od) in zip(axes, items):
        options = list(qd.iloc[0]["options"])
        w = qd["group_size"].to_numpy(float) if "group_size" in qd else np.ones(len(qd))
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(qd))
        P = pooled(qd, "truth", options, w)
        Qq = pooled(qd, "pred", options, w); Qo = pooled(od, "pred", options, w)
        x = np.arange(len(options))
        ax.bar(x - 0.27, P, 0.27, color="#6E6A62", label="human P")
        ax.bar(x, Qq, 0.27, color=AMBER, label="faithful @ Qwen")
        ax.bar(x + 0.27, Qo, 0.27, color=GREEN, label="our system")
        ax.set_xticks(x); ax.set_xticklabels([str(o)[:11] for o in options], fontsize=9, rotation=22, ha="right")
        ax.set_ylim(0, max(P.max(), Qq.max(), Qo.max()) * 1.32 + 1e-6)
        ax.set_title(f"{nice.get(q, q)}\nQwen S = {qd['score'].mean():.0f}   ·   ours S = {od['score'].mean():.0f}", fontsize=12)
        _apply(ax, c)
    axes[0].set_ylabel("probability"); axes[0].legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    save(fig, "15_required_q_qwen")
    for q, qd, od in items:
        print(f"    [req-Q] {q}: faithful@Qwen S={qd['score'].mean():.1f}  ·  ours S={od['score'].mean():.1f}")


def report_abstain():
    """Task C — how often the AbstainCalibrator falls back to uniform (test)."""
    from collections import Counter
    from scrye.data import load_all
    from scrye.splits import make_split
    df = frame(load("2026-06-21-decomp-final")["faithful"])  # faithful @ gemini-3.1 (final model)
    permean = df.groupby("dataset")["score"].mean()
    flagged = sorted(permean[permean < 0].index)  # exact abstain criterion: model worse than uniform
    allr = load_all(); full = allr["grouped"] + allr["pop"]
    test = make_split(full, seed=0, unit="question").test
    nfire = sum(1 for r in test if r.dataset_name in flagged)
    print("\n[ABSTAIN] flagged datasets (mean S < 0 on dev, faithful@3.1):")
    for ds in flagged:
        print(f"           {ds:16s} dev mean S = {permean[ds]:7.1f}")
    cnt = Counter(r.dataset_name for r in test if r.dataset_name in flagged)
    print(f"[ABSTAIN] test fire rate: {nfire}/{len(test)} = {100*nfire/len(test):.1f}% of items, on:")
    for ds, n in cnt.most_common():
        print(f"           {ds}: {n} test records")


def fig_prompt_cluster():
    # Most conditioning styles barely move off faithful; one breaks out.
    d = load("2026-06-20-baseline")
    c = C(False)
    order = [("faithful", "faithful\n(direct ask)"), ("contextualized", "contextualized"),
             ("representative_sample", "representative\nsample"), ("anti_flattening", "anti-\nflattening")]
    rows = []
    for key, lab in order:
        m, lo, hi = mean_ci([r["score"] for r in d[key]])
        rows.append((lab, m, lo, hi, key))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(rows)); fm = rows[0][1]
    for i, (lab, m, lo, hi, key) in enumerate(rows):
        ax.bar(i, m, 0.6, color=GREEN if key == "anti_flattening" else GREY)
        ax.plot([i, i], [lo, hi], color=INK, lw=1.6)
        ax.text(i, hi + 0.6, f"{m:.0f}", ha="center", fontsize=11, color=INK)
    ax.axhline(fm, color=SEC, lw=1.4, ls="--")
    ax.text(len(rows) - 0.5, fm + 0.5, "faithful baseline", ha="right", color=SEC, fontsize=10, style="italic")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_ylabel("SimBench score  S   (95% CI)")
    ax.set_ylim(0, 36)
    ax.set_title("Most conditioning styles ≈ the direct ask", fontsize=13)
    _apply(ax, c)
    save(fig, "10_prompt_cluster")


FIGS = [fig_score_hist, fig_score_vs_entropy, fig_examples, fig_calibration,
        fig_error_decomp, fig_graveyard_personas, fig_voting, fig_model_sweep,
        fig_headline_decomp, fig_result, fig_val_staircase, fig_cf_alignment,
        fig_required_q, fig_cross_task, fig_prompt_cluster]
# fig_required_q_qwen is intentionally NOT in FIGS: faithful@Qwen on the
# required-Q records is not in the LLM cache, so it would require live API.


def main():
    print(f"RUNS={RUNS}\nASSETS={ASSETS}\n")
    ok = 0
    for f in FIGS:
        try:
            f(); ok += 1
        except Exception:
            print(f"  ✗ {f.__name__} FAILED")
            traceback.print_exc()
    print(f"\n{ok}/{len(FIGS)} figures built")
    try:
        report_abstain()
    except Exception:
        print("  ✗ report_abstain FAILED")
        traceback.print_exc()


if __name__ == "__main__":
    main()
