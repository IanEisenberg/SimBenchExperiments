"""Assemble the results deck (docs/presentation/results-deck.html) — 19 slides on a
self-contained vanilla-JS viewer. Real figures (docs/presentation/assets/*.svg)
are inlined so the page's IBM Plex web fonts render their labels; conceptual
slides are authored as native HTML/SVG. No runtime, no build step to view.

    python scripts/build_deck.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "presentation" / "assets"
OUT = ROOT / "docs" / "presentation" / "results-deck.html"
TOT = 21

# palette
INK = "#1B1A17"; PAPER = "#F3EFE7"; SEC = "#8C8579"; GREEN = "#2E8568"
GREEN_L = "#5FCBA0"; AMBER = "#B5821E"; GREY = "#A39D92"; DARK = "#16151A"
LIGHTINK = "#ECE7DC"; CARD = "#FBF9F4"; LINE = "#DAD4C8"


def svg(name):
    return (ASSETS / f"{name}.svg").read_text()


def esc(s):
    return (s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;"))


def section(body, *, dark=False, label="", notes=""):
    cls = "sec dark" if dark else "sec"
    return (f'<section class="{cls}" data-label="{esc(label)}" '
            f'data-speaker-notes="{esc(notes)}">' + body + "</section>")


def hd(eyebrow, n=0):
    # page number is filled in sequentially at assembly (title slide has none)
    return (f'<div class="hd"><span class="eyebrow">{eyebrow}</span>'
            f'<span class="pageno">__PN__</span></div>')


def figwrap(name, maxh=600):
    return f'<div class="figwrap" style="--maxh:{maxh}px">' + svg(name) + "</div>"


def twofig(a, b, maxh=560):
    return ('<div class="figrow">'
            + f'<div class="figwrap" style="--maxh:{maxh}px">' + svg(a) + "</div>"
            + f'<div class="figwrap" style="--maxh:{maxh}px">' + svg(b) + "</div>"
            + "</div>")


def figslide(n, eyebrow, title, takeaway, fig, source, *, dark=False, label="", notes="", maxh=600):
    fig_html = fig if fig.lstrip().startswith("<") else figwrap(fig, maxh)
    body = (hd(eyebrow, n) + f'<h2 class="title">{title}</h2>'
            + (f'<div class="takeaway">{takeaway}</div>' if takeaway else "")
            + fig_html
            + f'<div class="source">{source}</div>')
    return section(body, dark=dark, label=label, notes=notes)


# =============================================================================
SLIDES = []

# --- 1 · Title --------------------------------------------------------------
bars = "".join(
    f'<div style="width:22px;height:{h}px;background:{GREEN_L};"></div>'
    for h in [16, 26, 42, 66, 96, 134, 178, 214, 214, 178, 134, 96, 66, 42, 26, 16])
SLIDES.append(section(
    f'''<div class="morph">{bars}</div>
    <div class="hd" style="position:relative;z-index:1;">
      <span class="eyebrow" style="color:#97928A;">Personal research · SimBench</span>
      <span class="pageno" style="color:#97928A;">Results deck</span></div>
    <div style="position:relative;z-index:1;margin-top:62px;">
      <h1 style="font-family:'Spectral',serif;font-weight:500;font-size:124px;line-height:.95;margin:0;letter-spacing:-.025em;white-space:nowrap;">SimBench Experiments</h1>
      <p style="font-family:'Spectral',serif;font-size:42px;line-height:1.3;margin:30px 0 0;max-width:1180px;color:#D8D2C6;">Predicting the distribution of human survey responses — and knowing which of our own wins are real.</p>
    </div>
    <div style="position:relative;z-index:1;margin-top:auto;display:flex;gap:26px;">
      <div class="statcard" style="background:rgba(95,203,160,.06);">
        <div class="statbig" style="color:{GREEN_L};">≈25 → ≈41</div>
        <div class="statsub">published SimBench baseline → our system<br><span style="color:#8C8579;">split-avg S · 0 = uniform, 100 = perfect</span></div></div>
      <div class="statcard"><div class="statbig"><span style="color:{GREEN_L};">+</span>10</div>
        <div class="statsub">better-fit base model<br><span style="color:#8C8579;">Qwen2.5-72B → gemini-3.1</span></div></div>
      <div class="statcard"><div class="statbig"><span style="color:{GREEN_L};">+</span>6</div>
        <div class="statsub">our method<br><span style="color:#8C8579;">on top, CI-clean</span></div></div>
    </div>
    <div style="position:relative;z-index:1;margin-top:26px;font-family:'IBM Plex Mono',monospace;font-size:23px;color:#97928A;">June 2026 — the deck is <em>how</em>, and how we know it's real.</div>''',
    dark=True, label="Title",
    notes="We took the published SimBench faithful baseline from about 25 to about 41 split-avg on a sealed test. Topline anatomy: +10 from a better-fit base model, +6 from our method. But the real contribution — and the talk — is how we got there and how we know the wins are real, not noise we talked ourselves into."))

# --- 2 · Task & metric ------------------------------------------------------
def minibars():
    P = [64, 128, 96, 40]; Q = [88, 112, 78, 52]
    cols = ""
    for p, q, lab in zip(P, Q, "ABCD"):
        cols += (f'<div style="display:flex;flex-direction:column;align-items:center;gap:8px;">'
                 f'<div style="display:flex;align-items:flex-end;gap:5px;height:150px;">'
                 f'<div style="width:22px;height:{p}px;background:{GREY};"></div>'
                 f'<div style="width:22px;height:{q}px;background:{GREEN};"></div></div>'
                 f'<div style="font-family:\'IBM Plex Mono\';font-size:22px;color:{SEC};">{lab}</div></div>')
    return cols

SLIDES.append(section(
    hd("01 · The task", 2) +
    '<h2 class="title">Predict the whole distribution, scored against "no information."</h2>'
    f'''<div style="display:flex;gap:60px;margin-top:36px;flex:1;min-height:0;">
      <div style="flex:1;display:flex;flex-direction:column;">
        <div class="kicker">INPUT</div>
        <div class="card"><div style="font-size:29px;line-height:1.4;">Survey question + discrete options + <span style="color:{SEC};">(optional)</span> demographic segment</div>
          <div style="font-family:'IBM Plex Mono';font-size:21px;color:{SEC};margin-top:8px;">segment may be empty = full population · no ground-truth access · must generalize to unseen questions</div></div>
        <div style="font-family:'Spectral';font-size:42px;color:{GREY};margin:8px 0;">↓</div>
        <div class="kicker">OUTPUT — predicted histogram <span style="color:{GREEN};">Q</span> vs human truth <span style="color:{GREY};">P</span></div>
        <div class="card" style="display:flex;align-items:flex-end;gap:30px;height:230px;">
          <div style="flex:1;display:flex;align-items:flex-end;justify-content:center;gap:22px;">{minibars()}</div>
          <div style="display:flex;flex-direction:column;gap:10px;font-family:'IBM Plex Mono';font-size:21px;">
            <span style="color:{SEC};">■ P human</span><span style="color:{GREEN};">■ Q predicted</span></div></div>
      </div>
      <div style="flex:1;display:flex;flex-direction:column;border-left:1px solid {LINE};padding-left:60px;">
        <div class="kicker">SCORED AGAINST THE UNIFORM GUESS</div>
        <div style="font-family:'Spectral';font-size:50px;margin-top:20px;line-height:1.2;">S = 100 × <span style="color:{SEC};">(</span>1 − <span style="color:{GREEN};">TVD(P,Q)</span> ⁄ <span style="color:{GREY};">TVD(P,U)</span><span style="color:{SEC};">)</span></div>
        <div style="font-size:26px;line-height:1.5;color:#46423B;margin-top:18px;">TVD (total variation distance) = ½·Σ|pᵢ − qᵢ| — the share of probability mass in the wrong place.</div>
        <div style="position:relative;height:120px;margin-top:40px;">
          <div style="position:absolute;left:0;right:0;top:64px;height:3px;background:{INK};"></div>
          <div style="position:absolute;left:0;top:52px;width:3px;height:26px;background:{INK};"></div>
          <div style="position:absolute;right:0;top:52px;width:3px;height:26px;background:{INK};"></div>
          <div style="position:absolute;left:0;top:84px;font-family:'IBM Plex Mono';font-size:22px;">0 · uniform</div>
          <div style="position:absolute;right:0;top:84px;font-family:'IBM Plex Mono';font-size:22px;">100 · perfect</div>
          <div style="position:absolute;left:40%;top:6px;transform:translateX(-50%);text-align:center;">
            <div style="font-family:'IBM Plex Mono';font-size:22px;color:{GREEN};">ours ≈ 41</div>
            <div style="width:18px;height:18px;background:{GREEN};transform:rotate(45deg);margin:6px auto 0;"></div></div>
        </div>
        <div style="margin-top:auto;display:inline-flex;align-self:flex-start;align-items:center;gap:12px;background:{CARD};border:1px solid {LINE};border-radius:30px;padding:11px 22px;">
          <span style="width:12px;height:12px;border-radius:50%;background:{INK};"></span>
          <span style="font-family:'IBM Plex Mono';font-size:22px;">3 pinned questions sealed into the test set</span></div>
      </div></div>'''
    + f'<div class="source">Source: SimBench (Hu et al. 2025) · <code>src/simbench_exp/scoring.py</code> · SimBench Eq. 2</div>',
    label="Task & metric",
    notes="The unit of truth is a histogram over options, not a single label. Score normalizes total-variation distance against the uniform guess: 0 means no better than 'everything equally likely', 100 a perfect match, negative worse than guessing. Three required questions are sealed into test."))

# --- Task types (the data) --------------------------------------------------
def tcard(name, share, datasets, desc, col):
    return (f'<div style="display:flex;gap:14px;border:1px solid {LINE};border-radius:8px;'
            f'background:{CARD};padding:15px 18px;">'
            f'<div style="width:6px;border-radius:3px;background:{col};flex:0 0 auto;"></div>'
            f'<div style="min-width:0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:10px;">'
            f'<span style="font-family:\'Spectral\',serif;font-size:27px;">{name}</span>'
            f'<span style="font-family:\'IBM Plex Mono\';font-size:17px;color:{SEC};white-space:nowrap;">{share}</span></div>'
            f'<div style="font-family:\'IBM Plex Mono\';font-size:16px;color:{GREEN};margin-top:6px;line-height:1.35;">{datasets}</div>'
            f'<div style="font-size:19px;color:#46423B;margin-top:5px;line-height:1.3;">{desc}</div>'
            f'</div></div>')

_comp = [(GREEN, 76.8), ("#6FA890", 6.5), (GREY, 4.7), ("#9DBBAF", 4.5), (AMBER, 3.8), ("#C99A3A", 3.7)]
_strip = "".join(f'<div style="flex:{w};background:{c};"></div>' for c, w in _comp)
SLIDES.append(section(
    hd("01 · The data") +
    '<h2 class="title">What\'s in SimBench — six kinds of question.</h2>'
    f'<div class="takeaway">Opinion surveys dominate (<b>77%</b>) and carry <b>all</b> the demographic segments; the rest are smaller, population-only behavioral &amp; knowledge tasks. Worth holding in mind — it\'s where the methods later diverge.</div>'
    f'<div style="display:flex;height:30px;border-radius:5px;overflow:hidden;margin-top:20px;border:1px solid {LINE};">{_strip}</div>'
    f'''<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px 24px;margin-top:24px;">
      {tcard("Opinion surveys","77% · pop + grouped","ESS · Afrobarometer · LatinoBarometro · OpinionQA · ISSP · GlobalOpinionQA · DICES · TISP","Attitudes &amp; beliefs — the only kind with demographic segments.",GREEN)}
      {tcard("Personality scales","6.5% · pop","OSPsych — MGKT · Big5 · MACH · RWAS","Psychometric self-report items.","#6FA890")}
      {tcard("Other","4.7% · pop","ChaosNLI · Jester","NLI disagreement · joke ratings.",GREY)}
      {tcard("Knowledge","4.5% · pop","NumberGame · WisdomOfCrowds","Estimation &amp; wisdom-of-crowds.","#9DBBAF")}
      {tcard("Moral dilemmas","3.8% · pop","MoralMachine","Trolley-style ethical choices — behavioral.",AMBER)}
      {tcard("Risky choices","3.7% · pop","Choices13k","Gambles &amp; lottery decisions — behavioral.","#C99A3A")}
    </div>'''
    + '<div class="source">Source: <code>simbench_exp.data.load_all</code> (13,510 records) · task-kind map</div>',
    label="The data",
    notes="Before any methods — what's actually in SimBench. Six kinds of question. Opinion surveys are 77% of it and the only kind with demographic segments — that's where we'll do well. The rest are small and population-only: personality scales, knowledge estimation, an 'other' bucket, and two behavioral kinds that'll matter later — moral dilemmas, which is MoralMachine, and risky choices, which is Choices13k."))

# --- 3 · Failure modes ------------------------------------------------------
SLIDES.append(figslide(
    0, "01 · Why it's hard",
    "How a predicted distribution goes wrong.",
    "Every wrong prediction splits into <b>two independent errors</b> that sum to the total (TVD): "
    "<b>concentration</b> — wrong spread (too sharp = mode-seeking, or too diffuse) — and "
    "<b>location</b> — right shape, mass on the <i>wrong</i> options. They trade off: sharpen to fix the "
    "mode and you over-concentrate; keep the spread and you blur it. A third, behavioral wrinkle — the "
    "<b>value–action gap</b>: gambles &amp; moral dilemmas resist survey-style elicitation (stated ≠ revealed).",
    "03_examples",
    "Source: <code>simbench_exp.decompose</code> on run <code>2026-06-21-decomp-final</code>",
    label="Failure modes", maxh=440,
    notes="Two independent error axes that sum to the total. Concentration is wrong spread — too sharp, which is mode-seeking, or too diffuse. Location is the right shape but mass on the wrong options. They trade off under any fix: sharpen the mode and you over-concentrate. And a third, behavioral wrinkle: the value-action gap — gambles and moral dilemmas resist survey-style elicitation."))

# --- 4 · The hypotheses -----------------------------------------------------
def hyp(name, verb, items):
    chips = "".join(f'<span class="chip">{it}</span>' for it in items)
    return (f'<div class="hypfam"><div class="hypfam-h">{name} '
            f'<span style="color:{GREY};font-weight:400;">· {verb}</span></div>'
            f'<div class="hypchips">{chips}</div></div>')
SLIDES.append(section(
    hd("01 · The hypotheses") +
    '<h2 class="title">So we wrote down everything we might try.</h2>'
    '<div class="takeaway">Most of these sound reasonable. On one dataset, anything can look good by chance — so <b>which are actually real?</b></div>'
    f'''<div class="hypgrid">
      {hyp("Reframe the ask", "prompting", ["faithful (paper baseline)", "anti-flattening / distributional", "contextualized", "representative-sample", "calibrated-commitment"])}
      {hyp("Make it reason", "multi-step", ["superforecaster CoT", "reasoning-first", "diversity-elicitation"])}
      {hyp("Simulate a population", "bottom-up", ["Monte-Carlo individuals", "persona embodiment", "representative persona sampling", "1M-persona census electorate (Nemotron)", "discrete voting / aggregation"])}
      {hyp("Fix the numbers after", "calibration", ["temperature scaling", "post-hoc calibrators", "entropy de-compression"])}
      {hyp("Route &amp; abstain", "meta", ["task-kind routing", "abstention (uniform fallback)", "task-context", "post-stratification"])}
      {hyp("Change the engine", "model", ["gemini-2.5 / 3.1 / 3.5-flash", "Qwen2.5-72B"])}
    </div>'''
    f'<div class="hypfoot">~18 plausible ideas · one dataset · the real adversary is <b style="color:{GREEN};">fooling yourself</b> → you need a method that won\'t.</div>'
    + '<div class="source">Source: <code>persona.py</code> (STRATEGIES) · <code>levers.py</code> · <code>docs/experiments/README.md</code> (Stages 01–18)</div>',
    label="The hypotheses",
    notes="So we wrote down everything we might try — about eighteen ideas in six families: reframe the prompt, make it reason, simulate a population bottom-up, fix the numbers after the fact, route and abstain, and change the base model. Most sound reasonable. But on a single dataset, anything can look good by chance. Which are actually real? That question forces the apparatus on the next slides."))

# --- 4 · Apparatus ----------------------------------------------------------
def pipebox(t, sub):
    return (f'<div class="pbox"><div class="pbox-t">{t}</div>'
            f'<div class="pbox-s">{sub}</div></div>')
arrow = '<div class="parrow">→</div>'
SLIDES.append(section(
    hd("02 · The apparatus") +
    '<h2 class="title">A method for finding methods — that doesn\'t fool itself.</h2>'
    f'<div class="takeaway">One idea = one subclass + a registry line. A <b>lever</b> swaps the <b>predictor</b>, the <b>calibrator</b>, or the <b>base model</b> — so every idea is tried cheaply and compared like-for-like.</div>'
    f'''<div style="margin-top:34px;">
      <div class="kicker" style="color:{GREEN_L};">THE SWAPPABLE SPINE</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:18px;flex-wrap:wrap;">
        {pipebox("Record","one typed survey row")}{arrow}{pipebox("Predictor","produces a distribution")}{arrow}{pipebox("Calibrator","post-hoc transform")}{arrow}{pipebox("score","SimBench S")}</div>
    </div>
    <div style="display:flex;gap:30px;margin-top:42px;flex:1;min-height:0;">
      <div class="defcard">
        <div class="defcard-h">Predictor</div>
        <div class="defcard-b">maps (question + options + optional demographic segment) → a predicted answer distribution.</div>
        <div class="defcard-e">e.g.&nbsp; zero-shot faithful ask&nbsp; ·&nbsp; calibrated-commitment prompt&nbsp; ·&nbsp; a bottom-up population simulation</div>
      </div>
      <div class="defcard">
        <div class="defcard-h">Calibrator</div>
        <div class="defcard-b">a post-hoc transform of that distribution — the predictor stays untouched.</div>
        <div class="defcard-e">e.g.&nbsp; identity (no-op)&nbsp; ·&nbsp; abstain → uniform on hard tasks&nbsp; ·&nbsp; temperature scaling</div>
      </div>
    </div>'''
    + '<div class="source">Source: <code>predict.py</code> · <code>calibrate.py</code> · <code>pipeline.py</code> · <code>experiment.py</code> (registry)</div>',
    dark=True, label="The apparatus",
    notes="The spine: a Record flows through a Predictor — which turns the question into a distribution — then a Calibrator, a post-hoc transform of that distribution, then scoring. A new idea is one subclass plus a registry line, and a lever can swap the predictor, the calibrator, or the base model. That's what lets us try eighteen ideas and compare them like-for-like."))

# --- 5 · Validation discipline ---------------------------------------------
def bucket(label, w, col, note):
    return (f'<div class="bkt"><div class="bkt-bar" style="background:{col};"></div>'
            f'<div class="bkt-l">{label}<span>{note}</span></div></div>')
SLIDES.append(section(
    hd("02 · The apparatus") +
    '<h2 class="title">Don\'t let the test leak; fix the rule before you look.</h2>'
    f'''<div style="display:flex;gap:60px;margin-top:28px;">
      <div style="flex:1;">
        <div class="kicker">SPLIT BY QUESTION FAMILY — never random rows</div>
        <div style="margin-top:18px;display:flex;flex-direction:column;gap:13px;">
          {bucket("dev","50","#3E9B77","tune freely · every call cached")}
          {bucket("val","25",AMBER,"η-gated · touched 3× (Stages 05/12/16)")}
          {bucket("test","25",GREY,"SEALED until the final run · required Qs pinned 🔒")}
        </div>
      </div>
      <div style="flex:1;border-left:1px solid {LINE};padding-left:60px;display:flex;flex-direction:column;justify-content:center;gap:18px;">
        <div class="vpoint"><b>Leave-family-out.</b> Split by <code>(dataset, input_template)</code> — a near-duplicate can't sit in two buckets.</div>
        <div class="vpoint"><b>Required Qs pinned to test.</b> trust_president · gay_rights · internet_use — never seen in tuning.</div>
        <div class="vpoint"><b>Preregistered.</b> The decision rule is fixed <i>before</i> the data is seen.</div>
      </div></div>
    <div style="margin-top:auto;border-top:1px solid {LINE};padding-top:22px;">
      <div class="kicker">THE GUARDED SEARCH LOOP</div>
      <div style="display:flex;align-items:center;gap:12px;margin-top:16px;flex-wrap:wrap;font-family:'IBM Plex Mono';font-size:22px;">
        <span class="loopstep">propose a lever</span><span class="looparr">→</span>
        <span class="loopstep">score on dev</span><span class="looparr">→</span>
        <span class="loopstep">dev-best? one η-gated val query</span><span class="looparr">→</span>
        <span class="loopstep" style="border-color:#E6CE97;background:#FBF1DC;">sealed test — once</span>
        <span style="color:{SEC};">· halts on budget / off-registry lever / gate fire</span>
      </div>
    </div>'''
    + '<div class="source">Source: <code>splits.py</code> · <code>search.py</code> · <code>ladder.py</code> · <code>manifest.py</code></div>',
    label="Validation discipline",
    notes="Three leakage guards plus the loop that ties them together. Split by question family so near-duplicates can't straddle buckets; pin the required questions to test; preregister the decision rule. Then the search loop: propose a lever, score it on dev, and only if it's dev-best does it earn a single eta-gated validation query — with the test sealed until the very end. It halts on budget, an off-registry lever, or a gate firing."))

# --- 6 · Ladder gate --------------------------------------------------------
SLIDES.append(figslide(
    6, "02 · The apparatus",
    "A val win counts only if it clears the noise floor.",
    "<b>Ladder gate (Blum &amp; Hardt 2015):</b> accept a validation improvement only if it beats the running best by more than <b>η</b>, a bootstrap-derived noise floor → a real bound on generalization error under adaptive reuse. Every step cleared its gate. <span style=\"color:#8C8579;\">(y-axis = <b>pooled validation S</b> — the record-weighted pop+grouped metric, same as the test \"overall\"; the router's val 47.0 → test 40.7 is the honest generalization gap.)</span>",
    "06_val_staircase",
    "Source: <code>ladder.py</code> · <code>results.py</code> · Stages 05 / 12 / 16",
    label="The Ladder gate", maxh=540,
    notes="This is the formal backbone. Blum and Hardt's Ladder: a validation result only counts if it beats the best so far by more than eta, a noise floor we derive from the bootstrap. That bounds generalization error even though we reuse validation adaptively. The staircase shows each accepted step."))

# --- 7 · Start naive --------------------------------------------------------
SLIDES.append(figslide(
    7, "03 · The journey",
    "Start naive: just ask the model.",
    "The honest baseline already scores in the 30s — but look <i>where</i> it fails: a worse-than-uniform tail, and scores that fall as questions get more divided. <b>Mode-seeking.</b>",
    twofig("07_score_hist", "07_score_vs_entropy", maxh=470),
    "Source: <code>simbench_exp.viz</code> on run <code>2026-06-21-decomp-final</code> (faithful @ gemini-3.1)",
    label="Start naive",
    notes="The first brick: just ask the model for the distribution. It's not terrible — mid-thirties — but the failure signature is clear. A worse-than-uniform tail, and scores that decline as the truth gets more divided. The model over-commits; it seeks the mode."))

# --- 8 · Graveyard I --------------------------------------------------------
SLIDES.append(figslide(
    8, "03 · The graveyard",
    "Simulating people loses to asking about people.",
    "Persona <b>embodiment</b> over-concentrates · <b>Monte-Carlo individuals −15.7</b> grouped · a <b>1M-persona census electorate</b> lands 51.8 vs the simple champion 63.2 (−11.4).",
    "08_graveyard_personas",
    "Source: Stage 03 (<code>persona-mechanisms</code>) · Stage 18 (<code>nemotron-personas</code>)",
    label="Graveyard I", maxh=500,
    notes="Act one of the graveyard. Every attempt to make the model more like a population of humans backfires. Embodying a persona over-concentrates. Sampling Monte-Carlo individuals loses fifteen points. A million-persona synthetic electorate underperforms a one-line prompt by eleven."))

# --- 9 · Graveyard II -------------------------------------------------------
SLIDES.append(figslide(
    9, "03 · The graveyard",
    "Most moves change nothing — a few are catastrophic.",
    "Most conditioning styles barely move off the direct ask (left); superforecaster / reasoning-first CoT lose outright. And discrete <b>voting</b> (right) is catastrophic where the mode is wrong — MoralMachine <b>−158</b>, its one transfer win didn't replicate. The search space is mostly <b>flat, with rare cliffs</b>.",
    twofig("10_prompt_cluster", "09_voting", maxh=430),
    "Source: Stages 01 / 07 / 13 (<code>voting-sim</code>)",
    label="Graveyard II",
    notes="Act two, reframed as the shape of the search space. On the left, most conditioning styles barely move off the direct ask — most things you try do nothing, and chain-of-thought variants actually lose. On the right, discrete voting is the cliff: it wins on one dataset but loses catastrophically on MoralMachine — minus 158 — by confidently committing to the wrong mode. Mostly flat, with rare cliffs. That's exactly why you need a disciplined search rather than trusting any single promising result."))

# --- 10 · Graveyard III -----------------------------------------------------
SLIDES.append(figslide(
    10, "03 · The graveyard",
    "You can't recalibrate your way out.",
    "Post-hoc calibration (temperature scaling, etc.) all lose to the <b>identity</b> calibrator. The model is over-confident — but it can't fix its own spread: entropy de-compression isn't recoverable from the model's own entropy (slope 0.48).",
    "10_calibration",
    "Source: Stages 06 / 08 / 09",
    label="Graveyard III", maxh=520,
    notes="Act three. If the model is miscalibrated, why not fix it after the fact? Because every post-hoc calibrator loses to doing nothing. The reliability plot shows the over-confidence, but the model can't predict its own error well enough to undo it — the recovery slope is under a half."))

# --- 11 · The turn ----------------------------------------------------------
def survivor(name, fixes, delta):
    return (f'<div class="surv"><div class="surv-n">{name}</div>'
            f'<div class="surv-f">{fixes}</div>'
            f'<div class="surv-d">{delta}</div></div>')
SLIDES.append(section(
    hd("04 · The turn", 11) +
    '<h2 class="title">What survived was simple — each a fix to a failure you just watched.</h2>'
    f'<div class="takeaway">No simulation, no chain-of-thought, no post-hoc calibration. Three principled moves — together the val-confirmed final system.</div>'
    f'''<div style="display:flex;gap:30px;margin-top:42px;flex:1;align-items:stretch;">
      {survivor("Distributional framing","Fixes over-concentration — keeps minority mass instead of collapsing to the mode.","+4.26 test grouped · val ✓")}
      {survivor("Calibrated-commitment prompt","Fixes <i>location</i> — gets the dominant option right, in one call.","+5.8 split-avg test (with abstention) · val ✓")}
      {survivor("Abstention","Knows when to abstain — falls back to uniform on the hardest ~5% of items instead of guessing.","the MVP — +3.26 pop val · val ✓")}
    </div>'''
    + '<div class="source">Source: Stages 05 / 10 / 11 / 12</div>',
    label="The turn",
    notes="Here's the turn. After all those failures, what held up was simple. Distributional framing preserves minority mass. A calibrated-commitment prompt fixes the location — the dominant option — in a single call. And abstention keeps us honest: where we reliably fail, fall back to uniform. Each one is a direct fix to a failure we just watched."))

# --- The winning prompts ----------------------------------------------------
SLIDES.append(section(
    hd("04 · The turn") +
    '<h2 class="title">The two prompts that did the work.</h2>'
    '<div class="takeaway">Both are one direct call — no simulation, no chain-of-thought. <b>Calibrated-commitment is anti-flattening plus one more move.</b></div>'
    f'''<div style="display:flex;gap:30px;margin-top:36px;flex:1;min-height:0;">
      <div class="pcard">
        <div class="pcard-h">Anti-flattening</div>
        <div class="pcard-tag">FIXES SPREAD</div>
        <div class="pcard-b">"Real groups disagree internally — <b>reproduce that heterogeneity</b>. Never flatten to one stereotyped answer; keep probability mass on the <b>minority views</b> that really exist. Estimate the <b>full distribution</b>, preserving the group's real internal spread."</div>
      </div>
      <div class="pcard">
        <div class="pcard-h">Calibrated-commitment <span style="font-family:'IBM Plex Mono';font-size:19px;color:{GREEN};">= anti-flattening + commit</span></div>
        <div class="pcard-tag">FIXES SPREAD + LOCATION</div>
        <div class="pcard-b">"Get two things right at once: <b>(1) which option leads</b> — get the mode right — and <b>(2) how divided</b> the group is. Keep minority mass, but <b>when one option clearly leads, give it a clear plurality</b> rather than hedging evenly. A broad population is usually more split than a subgroup."</div>
      </div>
    </div>'''
    + '<div class="source">Source: <code>src/simbench_exp/persona.py</code> — AntiFlattening / CalibratedCommitment system prompts (paraphrased)</div>',
    label="The winning prompts",
    notes="The two prompts that carried the method. Anti-flattening fixes spread: it tells the model real groups disagree and to keep mass on minority views, preserving the full distribution. Calibrated-commitment keeps all of that and adds one move — licensed commitment: when one option clearly leads, give it a clear plurality instead of hedging. That added move fixes location, the binding constraint. So calibrated-commitment is anti-flattening plus commit-to-the-mode."))

# --- The Nemotron bet -------------------------------------------------------
def nembar(label, v, col, vmax=66.0):
    h = max(int(v / vmax * 180), 2)
    return (f'<div style="display:flex;flex-direction:column;align-items:center;gap:8px;">'
            f'<div style="font-family:\'IBM Plex Mono\';font-size:21px;color:{INK};">{v}</div>'
            f'<div style="width:64px;height:{h}px;background:{col};border-radius:3px 3px 0 0;"></div>'
            f'<div style="font-family:\'IBM Plex Mono\';font-size:18px;color:{SEC};text-align:center;width:104px;line-height:1.2;">{label}</div></div>')
SLIDES.append(section(
    hd("04 · The bet we lost") +
    '<h2 class="title">The bet we spent the most on: a census-grounded electorate.</h2>'
    f'''<div style="display:flex;gap:56px;margin-top:28px;flex:1;min-height:0;">
      <div style="flex:1.15;" class="nemcol">
        <div class="kicker">WHY WE BET ON IT</div>
        <div style="margin-top:10px;">After the simple wins, gaps remained in <b>both location and spread</b>. We believed a <b>bottom-up simulation grounded in a real US population</b> — not the model's introspection — could capture the true structure.</div>
        <div class="kicker" style="margin-top:22px;">WHAT WE DID</div>
        <div style="margin-top:10px;"><b>Nemotron-Personas-USA</b> — 1M census-grounded synthetic US adults. A fixed <b>50-person representative panel</b> answers every OpinionQA question. Three rounds, each fixing the last failure:</div>
        <div style="margin-top:12px;display:flex;flex-direction:column;gap:7px;" class="nemstep">
          <div>① discrete vote per persona &nbsp;→&nbsp; over-concentrates (12.7)</div>
          <div>② per-persona distribution + average &nbsp;→&nbsp; <b style="color:{GREEN};">spread solved</b> (46.3 · entropy 0.64 vs truth 0.69)</div>
          <div>③ worldview + ideology enrichment &nbsp;→&nbsp; supplied the missing axis (51.8)</div>
        </div>
      </div>
      <div style="flex:1;border-left:1px solid {LINE};padding-left:56px;display:flex;flex-direction:column;">
        <div class="kicker">STILL LOST TO ONE DIRECT CALL &nbsp;<span style="color:{SEC};">· OpinionQA pop · dev</span></div>
        <div style="display:flex;align-items:flex-end;gap:22px;height:220px;margin-top:22px;">
          {nembar("discrete vote", 12.7, GREY)}{nembar("+ average", 46.3, GREEN_L)}{nembar("+ enriched", 51.8, GREEN_L)}{nembar("champion 1-call", 63.2, GREEN)}
        </div>
        <div style="margin-top:24px;font-size:23px;line-height:1.45;color:#46423B;"><b>Location-limited.</b> The 50 personas barely disagree — <b>78% pick the same option</b> — because Nemotron has no ideology/politics axis. Enrichment softens everyone to the center (74% moderate vs 37% real). Ensemble with the champion: <b>+0.47</b>, noise.</div>
      </div></div>'''
    + '<div class="source">Source: <code>docs/experiments/stage-18-nemotron-personas.md</code> — scores are OpinionQA-pop-dev (strongest slice, above the multi-dataset headline)</div>',
    label="The Nemotron bet",
    notes="This was the biggest single bet, and the one closest to how simulation is used in practice. After the simple wins, gaps remained in both location and spread, so we bet that a bottom-up simulation grounded in a real US population could beat introspection. Nemotron gives a million census-grounded synthetic adults; a fixed fifty-person panel answers every question. Three rounds each fixed the previous failure: discrete voting over-concentrated; per-persona distributions plus averaging solved spread entirely; worldview enrichment supplied the missing ideology axis. But it stayed location-limited — the personas barely disagree because Nemotron has no politics field — and even fully built it lost to a single calibrated-commitment call, with ensembling adding nothing. We closed the line. An honest, expensive negative."))

# --- 12 · Scoreboard --------------------------------------------------------
def row(lever, verdict, dot):
    return (f'<tr><td>{lever}</td><td><span class="dot" style="background:{dot};"></span>{verdict}</td></tr>')
SLIDES.append(section(
    hd("04 · The turn", 12) +
    '<h2 class="title">The receipt: what won, what didn\'t transfer, what lost.</h2>'
    f'''<table class="board">
      {row("Distributional framing (anti-flattening)", "won — +4.26 test grouped", GREEN)}
      {row("Base-model choice (gemini-3.1-flash-lite)", "won — dominant, +10 split-avg vs published baseline", GREEN)}
      {row("Calibrated-commitment prompt (fixes location)", "won — cc + abstain +5.8 split-avg test", GREEN)}
      {row("Abstention (uniform where we reliably fail)", "won — +3.26 pop val", GREEN)}
      {row("Task-kind routing", "won on val — <b>didn't transfer</b> (router = cc+abstain on test)", AMBER)}
      {row("Monte-Carlo personas · census electorate · voting · CoT · post-hoc calibration", "lose", GREY)}
    </table>
    <div style="display:flex;gap:34px;margin-top:30px;font-family:'IBM Plex Mono';font-size:23px;color:{SEC};">
      <span><span class="dot" style="background:{GREEN};"></span>won &amp; transferred</span>
      <span><span class="dot" style="background:{AMBER};"></span>won on val, didn't transfer</span>
      <span><span class="dot" style="background:{GREY};"></span>negative</span></div>'''
    + '<div class="source">Source: <code>docs/experiments/README.md</code> — stage table 01–18</div>',
    label="The scoreboard",
    notes="The full ledger — now the receipt, not the spoiler. A handful of real transferable wins. One honest amber: task-kind routing won on validation but tied the simpler system on test. And a column of instructive failures, each kept and reported."))

# --- 13 · Result ------------------------------------------------------------
SLIDES.append(figslide(
    13, "05 · The result",
    "A sealed test — and the paper reproduced.",
    "On a test set untouched until the final run, the method lifts faithful by <b>+5.8 split-avg</b> (CI excludes 0). And our harness brackets the published SimBench baseline — the prediction code is faithful.",
    "13_result",
    "Source: Stage 17 — <code>TEST-lineage</code> + <code>TEST-final</code> + <code>TEST-faithful-models</code>",
    dark=True, label="The result", maxh=540,
    notes="The payoff. On a sealed test set, holding the model fixed, our method adds about six points split-avg over faithful, and the confidence interval excludes zero — the wins are real, and the failures stayed dead. On the right, our faithful reproduction at Qwen brackets the paper's published number: the harness itself is faithful."))

# --- 14 · Model vs method ---------------------------------------------------
SLIDES.append(section(
    hd("05 · The result") +
    '<h2 class="title">The honest headline: the model did most of the work.</h2>'
    '<div class="takeaway">From the published baseline, <b>+10 is the base model · +6 is our method</b>. The model\'s gain is almost all on <b>grouped/demographic</b> questions (+18); our method carries <b>pop</b> (+6.7). And it\'s <b>fit, not size</b>: 3.1-flash-lite ≥ 3.5-flash everywhere, and cheaper.</div>'
    '<div class="callout">↳ <b>A local "bitter lesson"</b> (Sutton): the single biggest lever was a better-fit base model, not any one clever prompt. We report that out loud.</div>'
    '<div class="callout win">↳ <b>But +6 is no footnote</b> — ≈38% of the +16 total lift, and CI-clean. Holding the model fixed, the method still adds <b>+5.5 pooled / +6 split-avg</b> on the <b>sealed</b> test (distributional +4.26 grouped · abstention +3.26 pop). The model raised the ceiling; the method is the gap between strong and best.</div>'
    '<div class="figrow" style="margin-top:12px;">'
    '<div style="display:flex;flex-direction:column;align-items:center;flex:1;min-height:0;">'
    + figwrap("14_headline_decomp", 350) + '<div class="figcap">sealed test · split-avg</div></div>'
    '<div style="display:flex;flex-direction:column;align-items:center;flex:1;min-height:0;">'
    + figwrap("14_model_sweep", 350) + '<div class="figcap">dev · n=440 subsample · (we didn\'t rigorously sweep models)</div></div>'
    '</div>'
    + '<div class="source">Source: Stages 02 (<code>model-sweep</code>) / 17</div>',
    label="Model vs method",
    notes="The most honest slide — a local instance of the bitter lesson. Two-thirds of the gain over the published baseline is just a better-fit base model; one-third is our method. The model helps almost entirely on demographic questions; our method carries population. And it's fit not size — the cheaper 3.1-flash-lite matches or beats 3.5-flash everywhere. The left chart is the sealed test, the right is a dev subsample, and we did not exhaustively sweep models. But say the second half out loud too: that one-third is +6 of +16 — not a footnote. Holding the model fixed, the method still lifts a vanilla strong model by +5.5 pooled on the sealed test, every piece CI-clean — the model raised the ceiling, the method is the gap between strong and best."))

# --- 15 · Required questions ------------------------------------------------
SLIDES.append(figslide(
    15, "05 · The result",
    "The three required questions — first seen at evaluation.",
    "Pinned to test throughout. Predicted Q vs human P, with each question's SimBench S. Pooled required-Q progression: <b>−1.3</b> (faithful @ Qwen) → <b>+51.5</b> (model) → <b>+55.3</b> (our system).",
    "15_required_q",
    "Source: Stage 17 · notebook 03 — recomputed from cache (trust 56.5 · gay rights 48.2 · internet 61.4)",
    label="Required questions", maxh=480,
    notes="The three mandated questions, sealed until final evaluation. Predicted versus human, with each question's score. They go from worse-than-uniform at the paper baseline to the mid-fifties — most of that, again, the model swap, the rest our method."))

# --- 16 · Counterfactual ----------------------------------------------------
SLIDES.append(section(
    hd("05 · The result") +
    '<h2 class="title">Counterfactual sensitivity — measured, gated, honestly weak.</h2>'
    f'''<div style="display:flex;gap:60px;margin-top:28px;flex:1;min-height:0;">
      <div style="flex:1;display:flex;flex-direction:column;justify-content:center;">
        <div class="kicker">THE METHOD — a direction check, independent of accuracy</div>
        <div style="font-size:26px;line-height:1.5;margin-top:20px;color:#1B1A17;">
          <div>The <b>shift</b> = segment distribution − population distribution.</div>
          <div style="margin-top:14px;font-family:'IBM Plex Mono';font-size:24px;background:{CARD};border:1px solid {LINE};border-radius:6px;padding:14px 18px;"><b>cf_alignment</b> = cos( <span style="color:{GREEN};">predicted shift</span> , <span style="color:{GREY};">true shift</span> )</div>
          <div style="margin-top:14px;">+1 right direction · −1 wrong · 0 orthogonal · NaN when there's no shift.</div>
          <div style="margin-top:18px;color:#46423B;">Used as a <b>non-regression gate</b>: a system can't buy accuracy by moving demographics the <i>wrong</i> way. The signal is real but <b>weak</b>, and <b>our conditioning barely beats faithful</b> (right, +0.00 to +0.02) — directional sensitivity is mostly a property of the base model, not our method.</div>
        </div>
      </div>'''
    + figwrap("16_cf_alignment", 480)
    + '''</div>'''
    + '<div class="source">Source: <code>scoring.py</code> (<code>delta_alignment</code>) · Stages 01 / 04 / 05</div>',
    label="Counterfactual sensitivity",
    notes="The required counterfactual metric, explained. The shift is the segment distribution minus the population distribution. cf_alignment is the cosine between the predicted shift and the true shift: plus one is the right direction, minus one wrong, zero orthogonal. We use it as a non-regression gate so a system can't buy accuracy by getting directions wrong. Honestly the signal is weak and doesn't separate strategies, and it improves mostly with the model."))

# --- Cross-task / cross-group ----------------------------------------------
SLIDES.append(figslide(
    20, "05 · The result",
    "By task — our system vs the faithful baseline.",
    "Absolute SimBench S (<b>0 = uniform · 100 = perfect</b>, not a delta vs the paper) per task kind, faithful baseline → our final system. Our method <b>lifts every kind</b>: the biggest is the <b>abstention rescue on risky-choice</b> (−38 → ~0), then opinion surveys (the bulk, 46 → 51), personality, moral dilemmas, and 'other'; knowledge was already fine. Strongest where survey-style elicitation has signal. Smaller kinds carry wider CIs.",
    "18_cross_task",
    "Source: <code>simbench_exp.decompose</code> on <code>2026-06-21-decomp-final</code> (faithful vs final, dev) · task-kind map",
    label="By task", maxh=540,
    notes="Where does our method actually help? This is absolute SimBench S by task kind — zero is the uniform baseline, a hundred is perfect — for the faithful baseline versus our final system. Our method lifts every kind. The most dramatic is risky-choice, rescued from catastrophic, about minus thirty-eight, up to roughly uniform by abstention. Opinion surveys, the bulk of SimBench, are our strongest at around fifty. Knowledge was already fine and barely moves. These are absolute scores per kind, not a delta versus the paper."))

# --- 18 · Extensions + close ------------------------------------------------
def ext(name, answers):
    return (f'<div class="ext"><div class="ext-n">{name}</div>'
            f'<div class="ext-a">answers: {answers}</div></div>')
close_bars = "".join(
    f'<div style="width:18px;height:{h}px;background:{GREEN_L};opacity:{o};"></div>'
    for h, o in [(24, .45), (44, .6), (80, 1), (50, .6), (28, .45)])
SLIDES.append(section(
    hd("06 · What's next", 18) +
    '<h2 class="title">Each next step answers a failure we showed.</h2>'
    f'''<div style="display:flex;gap:64px;margin-top:34px;flex:1;min-height:0;">
      <div style="flex:1.1;display:flex;flex-direction:column;gap:16px;">
        {ext("Agentic tool-use (WebAgentPredictor)", "no real-world signal — leakage-guarded, must beat a closed-book twin and the champion")}
        {ext("Multi-agent critique / debate", "the persistent spread failure — surface minority mass, catch over-concentration")}
        {ext("More expressive multi-step prompting", "the graveyard — our naive CoT lost; promising but unproven")}
        {ext("Fuller, principled model sweep", "the model-vs-method finding — the dominant lever, unexplored")}
      </div>
      <div style="flex:1;display:flex;flex-direction:column;background:{DARK};border-radius:10px;padding:34px 36px;color:{LIGHTINK};">
        <div class="kicker" style="color:{GREEN_L};">THE CLOSE</div>
        <div style="font-size:27px;line-height:1.5;color:#D8D2C6;margin-top:16px;">Beat the baseline on a sealed test, reproduced the paper, demoted the wins that didn't transfer. <span style="color:{LIGHTINK};">The apparatus generalizes to any LLM-eval problem.</span></div>
        <div style="margin-top:auto;display:flex;align-items:flex-end;gap:12px;height:90px;opacity:.9;">{close_bars}</div>
        <div style="font-family:'Spectral';font-size:29px;line-height:1.25;color:{GREEN_L};margin-top:18px;">→ Open question: from matching survey marginals to predicting what people actually do.</div>
      </div></div>'''
    + '<div class="source">Source: Stages 13 / 16 / 18 · SimBench (Hu et al. 2025)</div>',
    dark=True, label="Extensions & close",
    notes="Future work isn't a wishlist — each item answers a failure from this talk. Agentic tool-use for the missing real-world signal. Multi-agent critique for the spread problem. More structured multi-step prompting, honestly flagged because our naive version lost. And a real model sweep, since the model is the dominant lever and we never explored it. The transferable asset is the discipline. The open question the literature still leaves is the value-action gap: can a simulator that matches stated survey answers also predict revealed behavior — the gambles and moral dilemmas where we abstained?"))


# =============================================================================
HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SimBench Experiments — Results Deck</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  html,body{margin:0;padding:0;height:100%;background:#16151A;}
  *{box-sizing:border-box;}
  #viewport{position:fixed;inset:0;overflow:hidden;display:flex;align-items:center;justify-content:center;background:#16151A;}
  #stage{width:1920px;height:1080px;position:relative;flex:0 0 auto;transform-origin:center center;box-shadow:0 30px 120px rgba(0,0,0,.55);}
  .slide{position:absolute;inset:0;display:none;}.slide.active{display:block;}
  .slide>section{width:100%;height:100%;}
  /* slide frame */
  section.sec{height:100%;padding:78px 104px 60px;background:#F3EFE7;color:#1B1A17;font-family:'IBM Plex Sans',system-ui,sans-serif;display:flex;flex-direction:column;position:relative;overflow:hidden;}
  section.sec.dark{background:#16151A;color:#ECE7DC;}
  .hd{display:flex;justify-content:space-between;align-items:baseline;}
  .eyebrow{font-family:'IBM Plex Mono',monospace;font-size:25px;letter-spacing:.16em;text-transform:uppercase;color:#8C8579;}
  .pageno{font-family:'IBM Plex Mono',monospace;font-size:25px;letter-spacing:.1em;color:#8C8579;}
  h2.title{font-family:'Spectral',Georgia,serif;font-weight:500;font-size:66px;line-height:1.05;margin:16px 0 0;letter-spacing:-.01em;max-width:1660px;}
  .dark h2.title{color:#ECE7DC;}
  .takeaway{font-size:29px;line-height:1.42;color:#46423B;margin:18px 0 0;max-width:1640px;}
  .dark .takeaway{color:#C9C3B7;}
  .takeaway b{color:#2E8568;font-weight:600;} .dark .takeaway b{color:#5FCBA0;}
  .source{margin-top:auto;font-family:'IBM Plex Mono',monospace;font-size:22px;color:#A39D92;padding-top:16px;}
  .source code,.takeaway code{font-family:'IBM Plex Mono',monospace;font-size:.92em;}
  .figwrap{flex:1;display:flex;align-items:center;justify-content:center;min-height:0;margin-top:12px;}
  .figwrap svg{max-width:100%;max-height:var(--maxh,580px);height:auto;width:auto;}
  .figrow{flex:1;display:flex;gap:40px;align-items:center;justify-content:center;min-height:0;margin-top:8px;}
  .kicker{font-family:'IBM Plex Mono',monospace;font-size:23px;letter-spacing:.1em;color:#8C8579;}
  .card{border:1px solid #DAD4C8;border-radius:6px;background:#FBF9F4;padding:22px 26px;margin-top:12px;}
  /* title */
  .morph{position:absolute;left:0;right:0;bottom:0;height:300px;display:flex;align-items:flex-end;justify-content:center;gap:15px;opacity:.13;pointer-events:none;}
  .statcard{flex:1;border:1px solid rgba(236,231,220,.18);border-radius:5px;padding:26px 30px;}
  .statbig{font-family:'Spectral',serif;font-weight:500;font-size:78px;line-height:1;color:#ECE7DC;}
  .statsub{font-size:24px;line-height:1.4;color:#B7B2A8;margin-top:10px;}
  /* apparatus */
  .pbox{border:1px solid rgba(236,231,220,.2);border-radius:7px;padding:18px 22px;background:rgba(95,203,160,.05);}
  .pbox-t{font-family:'Spectral',serif;font-size:34px;color:#ECE7DC;} .pbox-s{font-family:'IBM Plex Mono';font-size:19px;color:#97928A;margin-top:4px;}
  .parrow{font-size:34px;color:#5FCBA0;margin:0 4px;}
  .frow{display:flex;align-items:center;gap:18px;} .frow span{font-family:'IBM Plex Mono';font-size:22px;color:#C9C3B7;white-space:nowrap;}
  .fbar{height:30px;border-radius:4px;}
  /* validation buckets */
  .bkt{display:flex;align-items:center;gap:20px;} .bkt-bar{width:90px;height:54px;border-radius:5px;flex:0 0 auto;}
  .bkt-l{font-family:'Spectral',serif;font-size:38px;} .bkt-l span{display:block;font-family:'IBM Plex Sans';font-size:22px;color:#8C8579;margin-top:2px;}
  .vpoint{font-size:27px;line-height:1.4;} .vpoint b{color:#2E8568;}
  /* survivors */
  .surv{flex:1;border:1px solid #DAD4C8;border-radius:9px;background:#FBF9F4;padding:30px 30px;display:flex;flex-direction:column;border-top:5px solid #2E8568;}
  .surv-n{font-family:'Spectral',serif;font-size:40px;line-height:1.1;} .surv-f{font-size:25px;line-height:1.45;color:#46423B;margin-top:16px;flex:1;}
  .surv-d{font-family:'IBM Plex Mono';font-size:23px;color:#2E8568;margin-top:18px;}
  /* scoreboard */
  table.board{width:100%;border-collapse:collapse;margin-top:30px;font-size:28px;}
  table.board td{padding:18px 8px;border-bottom:1px solid #E4DECF;vertical-align:top;}
  table.board td:first-child{width:56%;} table.board b{color:#1B1A17;}
  .dot{display:inline-block;width:16px;height:16px;border-radius:50%;margin-right:14px;vertical-align:middle;}
  /* extensions */
  .ext{border:1px solid #DAD4C8;border-left:5px solid #2E8568;border-radius:7px;background:#FBF9F4;padding:20px 26px;}
  .ext-n{font-family:'Spectral',serif;font-size:32px;color:#1B1A17;} .ext-a{font-size:22px;color:#46423B;margin-top:6px;}
  /* apparatus definition cards */
  .defcard{flex:1;border:1px solid rgba(236,231,220,.2);border-radius:9px;background:rgba(95,203,160,.05);padding:28px 32px;display:flex;flex-direction:column;}
  .defcard-h{font-family:'Spectral',serif;font-size:38px;color:#ECE7DC;}
  .defcard-b{font-size:25px;line-height:1.45;color:#C9C3B7;margin-top:12px;}
  .defcard-e{font-family:'IBM Plex Mono',monospace;font-size:20px;color:#97928A;margin-top:auto;padding-top:18px;}
  /* search loop */
  .loopstep{background:#FBF9F4;border:1px solid #DAD4C8;border-radius:6px;padding:8px 16px;color:#1B1A17;white-space:nowrap;}
  .looparr{color:#8C8579;}
  /* prompt cards */
  .pcard{flex:1;border:1px solid #DAD4C8;border-radius:9px;background:#FBF9F4;padding:26px 30px;display:flex;flex-direction:column;border-top:5px solid #2E8568;}
  .pcard-h{font-family:'Spectral',serif;font-size:34px;}
  .pcard-tag{font-family:'IBM Plex Mono',monospace;font-size:19px;letter-spacing:.06em;color:#2E8568;margin-top:4px;}
  .pcard-b{font-size:23px;line-height:1.5;color:#46423B;margin-top:16px;}
  .pcard-b b{color:#1B1A17;}
  /* figure captions + callout */
  .figcap{font-family:'IBM Plex Mono',monospace;font-size:20px;color:#8C8579;text-align:center;margin-top:6px;}
  .callout{margin-top:16px;background:#FBF1DC;border:1px solid #E6CE97;border-radius:8px;padding:15px 22px;font-size:25px;line-height:1.4;color:#5A4A1E;}
  .callout b{color:#3E3411;}
  .callout.win{margin-top:12px;background:#E7F3EE;border-color:#A9D8C4;color:#1E5A45;}
  .callout.win b{color:#16432F;}
  /* nemotron */
  .nemcol{font-size:25px;line-height:1.5;color:#46423B;} .nemcol b{color:#1B1A17;}
  .nemstep{font-family:'IBM Plex Mono',monospace;font-size:21px;color:#46423B;}
  /* hypotheses */
  .hypgrid{flex:1;display:grid;grid-template-columns:repeat(3,1fr);gap:22px 30px;margin-top:26px;min-height:0;align-content:start;}
  .hypfam{display:flex;flex-direction:column;}
  .hypfam-h{font-family:'IBM Plex Mono',monospace;font-size:21px;letter-spacing:.04em;color:#1B1A17;font-weight:500;margin-bottom:13px;text-transform:uppercase;}
  .hypchips{display:flex;flex-wrap:wrap;gap:8px;}
  .chip{font-family:'IBM Plex Sans',sans-serif;font-size:20px;color:#46423B;background:#FBF9F4;border:1px solid #DAD4C8;border-radius:20px;padding:6px 14px;line-height:1.2;}
  .hypfoot{margin-top:16px;font-family:'IBM Plex Mono',monospace;font-size:23px;color:#8C8579;border-top:1px solid #E4DECF;padding-top:16px;}
  .hypfoot b{font-weight:600;}
  /* chrome */
  #hud{position:fixed;right:22px;bottom:18px;z-index:50;font-family:'IBM Plex Mono',monospace;font-size:15px;color:#9a948a;background:rgba(22,21,26,.72);border:1px solid rgba(236,231,220,.18);border-radius:30px;padding:7px 16px;}
  #help{position:fixed;left:22px;bottom:18px;z-index:50;font-family:'IBM Plex Mono',monospace;font-size:13px;color:#6f6a62;}
  .navzone{position:fixed;top:0;bottom:0;width:12%;z-index:40;cursor:pointer;}#navprev{left:0;}#navnext{right:0;}
  #notes{position:fixed;left:0;right:0;bottom:0;z-index:60;max-height:44vh;overflow:auto;background:rgba(15,14,18,.97);color:#D8D2C6;border-top:1px solid rgba(236,231,220,.2);padding:20px 30px 26px;font-family:'IBM Plex Sans';font-size:19px;line-height:1.55;display:none;}#notes.show{display:block;}
  #notes .lbl{font-family:'IBM Plex Mono',monospace;font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#7d776e;margin-bottom:9px;}
  @media print{@page{size:1920px 1080px;margin:0;}html,body{background:#fff;height:auto;}#viewport{position:static;display:block;overflow:visible;}#stage{transform:none!important;box-shadow:none;width:1920px;height:1080px;}.slide{position:relative;display:block!important;width:1920px;height:1080px;page-break-after:always;break-after:page;}.slide:last-child{page-break-after:auto;}#hud,#help,.navzone,#notes{display:none!important;}}
</style>
</head>
<body>
<div id="viewport"><div id="stage">
"""

SCRIPT = """
</div></div>
<div class="navzone" id="navprev"></div><div class="navzone" id="navnext"></div>
<div id="hud"><span id="cur">1</span> / <span id="tot">%d</span></div>
<div id="help">&larr; &rarr; slides &middot; N notes &middot; F fullscreen</div>
<div id="notes"><div class="lbl">Speaker notes</div><div id="notesbody"></div></div>
<script>
var stage=document.getElementById('stage'),slides=[].slice.call(stage.querySelectorAll('.slide')),tot=slides.length,i=0;
document.getElementById('tot').textContent=tot;
function fit(){var s=Math.min(innerWidth/1920,innerHeight/1080);stage.style.transform='scale('+s+')';}
function show(n){i=Math.max(0,Math.min(tot-1,n));for(var k=0;k<tot;k++)slides[k].classList.toggle('active',k===i);
 document.getElementById('cur').textContent=i+1;var s=slides[i].querySelector('section');
 document.getElementById('notesbody').textContent=(s&&s.getAttribute('data-speaker-notes'))||'(no notes)';
 var h='#'+(i+1);if(location.hash!==h)history.replaceState(null,'',h);}
function next(){show(i+1);}function prev(){show(i-1);}
addEventListener('resize',fit);
addEventListener('keydown',function(e){var k=e.key;
 if(k==='ArrowRight'||k===' '||k==='PageDown'||k==='ArrowDown'){e.preventDefault();next();}
 else if(k==='ArrowLeft'||k==='PageUp'||k==='ArrowUp'){e.preventDefault();prev();}
 else if(k==='Home'){show(0);}else if(k==='End'){show(tot-1);}
 else if(k==='n'||k==='N'){document.getElementById('notes').classList.toggle('show');}
 else if(k==='f'||k==='F'){if(!document.fullscreenElement){document.documentElement.requestFullscreen&&document.documentElement.requestFullscreen();}else{document.exitFullscreen&&document.exitFullscreen();}}
 else if(/^[0-9]$/.test(k)){var d=k==='0'?10:parseInt(k,10);if(d>=1&&d<=tot)show(d-1);}});
document.getElementById('navnext').addEventListener('click',next);
document.getElementById('navprev').addEventListener('click',prev);
var st=parseInt((location.hash||'').slice(1),10);if(!(st>=1&&st<=tot))st=1;fit();show(st-1);
</script>
</body></html>"""

import re
TOT = len(SLIDES)  # auto — page-number total never drifts when slides are added
html = HEAD + "\n".join(f'<div class="slide">{s}</div>' for s in SLIDES) + (SCRIPT % TOT)
_pn = [1]
def _num(_m):
    _pn[0] += 1
    return f"{_pn[0]:02d} / {TOT}"
html = re.sub("__PN__", _num, html)            # number content slides 02..TOT in order
assert "__PN__" not in html, "unfilled page-number placeholder"
OUT.write_text(html)
print(f"wrote {OUT} ({len(html)} bytes, {len(SLIDES)} slides)")
