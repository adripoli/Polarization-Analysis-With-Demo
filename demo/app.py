# demo/app.py
# live demo of the polarization pipeline. one tweet goes through every version
# (Marks 1-3 are the actual old code from git, Mark 4 is a retrain of e554bad,
# Final is the model in ../model) so you can watch each one get it right or wrong.
#
#   python demo/app.py                 -> opens the web demo at http://127.0.0.1:7860
#   python demo/app.py --share         -> same, plus a temporary public link
#   python demo/app.py --cli "tweet"   -> no browser, just prints every Mark's call

import argparse, importlib, importlib.util, json, os, random, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)  # so the final pipeline's own model.py / config.py import cleanly

MARK_INFO = {
    "Mark 1": "keyword match (commit 1ba92a1)",
    "Mark 2": "+ negative adjectives flip the score (22f45be)",
    "Mark 3": "+ adjectives weighted by distance to the keyword (a2f78b5)",
    "Mark 4": "fine-tuned BERTweet on ~730 hand labels (e554bad, retrained)",
    "Final": "BERTweet + weak labels + hand labels x5 (saved final model)",
}


# ── Marks 1-3: load each commit's code side by side ────────────────────────
# they all do `from keywords import ...`, so import one, grab what we need,
# then wipe the module cache before the next one or they'd share lists
def _load_rules(name):
    d = os.path.join(HERE, "marks", name)
    for m in ("analyze_tweets", "keywords", "adjectives"):
        sys.modules.pop(m, None)
    sys.path.insert(0, d)
    at = importlib.import_module("analyze_tweets")
    kw = importlib.import_module("keywords")
    sys.path.pop(0)
    for m in ("analyze_tweets", "keywords", "adjectives"):
        sys.modules.pop(m, None)
    return at.score_tweet, kw.TRUMP_KEYWORDS + kw.BIDEN_KEYWORDS


RULES = {f"Mark {i}": _load_rules(f"mark{i}") for i in (1, 2, 3)}


# ── Mark 4 + Final: the two BERTweet models (loaded lazily, they're big) ───
_models = {}


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mark4():
    if "m4" not in _models:
        import torch
        spec = importlib.util.spec_from_file_location("mark4_model", os.path.join(HERE, "marks", "mark4", "model.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        net = mod.BERTweetStanceClassifier()
        net.load_state_dict(torch.load(os.path.join(ROOT, "model", "mark4_rerun", "mark4_seed1.pt"), map_location="cpu"))
        _models["m4"] = (net.to(_device()).eval(), mod)
    return _models["m4"]


def _final():
    if "fin" not in _models:
        import torch
        import model as fm  # the final pipeline's model.py
        net = fm.BERTweetStanceClassifier()
        net.load_state_dict(torch.load(os.path.join(ROOT, "model", "bertweet_finetuned", "best_model.pt"), map_location="cpu"))
        _models["fin"] = (net.to(_device()).eval(), fm)
    return _models["fin"]


def _clean_mark4(t):
    # same cleaning create_manual_labels.py used for Mark 4's training data
    t = re.sub(r"http\S+|www\S+", "", str(t)); t = re.sub(r"@\w+", "", t); t = t.replace("#", "")
    return re.sub(r"\s+", " ", t).strip()


def _clean_final(t):
    from stage0_cleaning import clean_text  # the final pipeline's stage 0
    return clean_text(str(t))


def model_probs(which, texts):
    """P(Biden), P(Trump) for a list of tweets. Final also has a spam column
    but that head collapsed in the saved checkpoint, so it's ignored here and
    the stance is renormalised over the two political classes like the pipeline does"""
    import torch
    net, mod = _mark4() if which == "Mark 4" else _final()
    clean = _clean_mark4 if which == "Mark 4" else _clean_final
    tok = mod.load_tokenizer()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = mod.tokenize_batch([clean(t) or "." for t in texts[i:i + 32]], tok)
            p = net(enc["input_ids"].to(_device()), enc["attention_mask"].to(_device()))["probs"].float().cpu().numpy()
            pol = p[:, :2] / p[:, :2].sum(1, keepdims=True)
            out.append(pol)
    return np.vstack(out)


# ── run one tweet through everything ──────────────────────────────────────
def all_marks(text):
    res = {}
    low = str(text).lower()
    for m, (fn, kws) in RULES.items():
        s = fn(str(text))
        hits = sorted({k for k in kws if k in low})
        res[m] = dict(side={1: "Trump", -1: "Biden", 0: "no call"}[s], conf=None,
                      why=("keywords: " + ", ".join(hits)) if hits else "no keyword found")
    for m in ("Mark 4", "Final"):
        pb, pt = model_probs(m, [text])[0]
        res[m] = dict(side="Trump" if pt > pb else "Biden", conf=float(max(pb, pt)),
                      why=(f"P(Trump) = {pt:.3f}, P(Biden) = {pb:.3f}" if abs(pt - pb) < 0.05
                           else f"P(Trump) = {pt:.2f}, P(Biden) = {pb:.2f}"))
    return res


# ── HTML bits ─────────────────────────────────────────────────────────────
COL = {"Trump": "#e34948", "Biden": "#2a78d6", "no call": "#9a9993", "spam": "#6250d6"}
CSS = """
.pp-card{border:1px solid #e4e3df;border-radius:10px;padding:10px 14px;margin:6px 0;background:#fcfcfb;
         display:flex;align-items:center;gap:14px}
.pp-name{font-weight:700;min-width:70px;color:#0b0b0b}
.pp-desc{color:#52514e;font-size:13px;flex:1}
.pp-chip{color:white;border-radius:999px;padding:3px 12px;font-weight:700;font-size:13px;min-width:70px;text-align:center}
.pp-why{color:#52514e;font-size:12px;min-width:210px;text-align:right}
.pp-ok{font-size:18px;min-width:20px;text-align:center}
.pp-tweet{font-size:15px;padding:10px 14px;border-left:4px solid #2a78d6;background:#f3f2ee;border-radius:6px;margin-bottom:6px}
"""


def cards(res, truth=None):
    html = ""
    for m, r in res.items():
        mark = ""
        if truth in ("Trump", "Biden"):
            mark = "✅" if r["side"] == truth else "❌"
        html += (f"<div class='pp-card'><div class='pp-name'>{m}</div><div class='pp-desc'>{MARK_INFO[m]}</div>"
                 f"<div class='pp-why'>{r['why']}</div>"
                 f"<div class='pp-chip' style='background:{COL[r['side']]}'>{r['side']}</div>"
                 f"<div class='pp-ok'>{mark}</div></div>")
    return html


def tab_one(text):
    if not str(text).strip():
        return "<i>Type a tweet first.</i>"
    return cards(all_marks(text))


# ── tab 2: score a whole user with the final formula ──────────────────────
POP = pd.read_csv(os.path.join(HERE, "data", "population.csv"))
MAX_STD = 0.8017722964286804  # biggest per-user spread in the full 2nd run


def tab_user(block):
    tweets = [l.strip() for l in str(block).splitlines() if l.strip()]
    if len(tweets) < 2:
        return "<i>Paste at least 2 tweets, one per line.</i>", None
    p = model_probs("Final", tweets)
    stance = p[:, 1] - p[:, 0]
    ent = -(p * np.log2(np.clip(p, 1e-12, 1))).sum(1)  # same binary entropy as stage4_5
    mean_entropy = 1 - ent.mean()
    consistency = max((stance > 0).sum(), (stance < 0).sum()) / len(stance)
    std = float(np.std(stance))
    raw = 0.6 * (1 - min(std / MAX_STD, 1)) + 0.2 * mean_entropy + 0.2 * consistency
    pct = (POP.polarization_raw < raw).mean() * 100
    rows = "".join(f"<tr><td style='padding:3px 8px'>{t[:110]}</td><td style='padding:3px 8px;color:{COL['Trump' if s > 0 else 'Biden']};"
                   f"font-weight:700'>{s:+.2f}</td></tr>" for t, s in zip(tweets, stance))
    dup = len(set(tweets)) == 1
    warn = ("<p style='color:#b3501f'><b>Every tweet is identical.</b> Spread is zero, so Mark 4 would rank this account "
            "as maximally polarized. Mark 5's dedupe collapses it to 1 tweet and it drops out. And notice the model "
            "still picks a side for what is basically an ad: it has no working spam detector, which is the next "
            "thing on the list.</p>") if dup else ""
    html = (f"{warn}<p><b>Mean stance:</b> {stance.mean():+.2f} &nbsp; <b>Spread (std):</b> {std:.2f} &nbsp; "
            f"<b>Consistency:</b> {consistency:.0%} &nbsp; <b>Model confidence:</b> {mean_entropy:.2f}</p>"
            f"<p style='font-size:18px'><b>Polarization score: {raw:.2f}</b> &nbsp;→&nbsp; more polarized than "
            f"<b>{pct:.0f}%</b> of the 30,525 real users from the full run</p>"
            f"<table style='font-size:13px'><tr><th align=left>tweet</th><th>stance</th></tr>{rows}</table>")
    return html, _pop_plot(float(stance.mean()))


def _pop_plot(user_mean):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.hist(POP.mean_stance, bins=70, range=(-1, 1), color="#dcdbd5", edgecolor="#fcfcfb", linewidth=0.4)
    ax.axvline(user_mean, color=COL["Trump" if user_mean > 0 else "Biden"], lw=3)
    ax.text(user_mean, ax.get_ylim()[1] * 0.92, "  this user", color="#0b0b0b", fontsize=9,
            ha="left" if user_mean < 0.6 else "right")
    ax.set_xlabel("mean stance  (-1 = pro-Biden, +1 = pro-Trump)"); ax.set_yticks([])
    ax.set_title("Where they'd sit among the 30,525 users", loc="left", fontsize=10, fontweight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# ── tab 3: grade it yourself on real held-out tweets ──────────────────────
HELD = pd.read_csv(os.path.join(HERE, "data", "heldout.csv"))
HELD = HELD[HELD.label != 2].reset_index(drop=True)  # political ones only, same as the report
BOARD = json.load(open(os.path.join(HERE, "data", "scoreboard.json")))


def tab_grade(tally):
    row = HELD.iloc[random.randrange(len(HELD))]
    truth = "Trump" if row.label == 1 else "Biden"
    res = all_marks(row.cleaned_text)
    tally = dict(tally or {}); tally["n"] = tally.get("n", 0) + 1
    for m, r in res.items():
        tally[m] = tally.get(m, 0) + (r["side"] == truth)
    head = (f"<div class='pp-tweet'>{row.cleaned_text}</div>"
            f"<p>My hand label: <b style='color:{COL[truth]}'>{truth}</b></p>")
    score = " &nbsp;·&nbsp; ".join(f"<b>{m}</b> {tally[m]}/{tally['n']}" for m in res)
    return head + cards(res, truth) + f"<p style='margin-top:10px'>Your running tally: {score}</p>", tally


def scoreboard_md():
    lines = ["| Version | Correct on 155 held-out tweets | No call |", "|---|---|---|"]
    for b in BOARD:
        lines.append(f"| {b['mark']} ({b['commit']}) | **{b['acc']*100:.0f}%** | {b['nocall']*100:.0f}% |")
    return "\n".join(lines)


EXAMPLES = [
    "Those MAGA people are crazy",
    "Incompetent. Just like everyone else involved in the trump clown car. VoteBiden",
    "trump is crazy good at rallies",
    "Ugh, of course the MAGA crowd thinks this is brilliant. Great job guys. VoteBiden",
    "Four more years! Best economy ever, nobody has done more for this country #Trump2020",
    "Vote him out. We can't take four more years of this chaos. #BidenHarris2020",
]
USER_EXAMPLES = [
    "Four more years! #MAGA\nBiden can't even remember what state he's in\nThe economy was booming before covid, give him another term\n"
    "Sleepy Joe and the radical left want to defund the police",
    "Vote him out #BidenHarris2020\nTrump lied about covid and people died\nCan't wait for a president who believes in science\n"
    "Early voted for Biden today!",
    "Get your official election 2020 shirts now at our store!\nGet your official election 2020 shirts now at our store!\n"
    "Get your official election 2020 shirts now at our store!\nGet your official election 2020 shirts now at our store!",
]


def build_ui():
    import gradio as gr
    with gr.Blocks(title="Polarization Pipeline demo") as ui:
        gr.Markdown("# Polarization Pipeline: live demo\nOne tweet through every version I built. Marks 1-3 are the "
                    "real old code from git; Mark 4 and Final are BERTweet models.")
        with gr.Tab("1. One tweet, every Mark"):
            box = gr.Textbox(label="Tweet", lines=2, placeholder="Type any election tweet...")
            go = gr.Button("Run through every Mark", variant="primary")
            out = gr.HTML()
            gr.Examples(EXAMPLES, inputs=box, label="Try the tweets from the write-up")
            go.click(tab_one, box, out); box.submit(tab_one, box, out)
        with gr.Tab("2. Score a user"):
            ubox = gr.Textbox(label="A user's tweets, one per line", lines=6)
            ugo = gr.Button("Score this user", variant="primary")
            uout = gr.HTML(); uplot = gr.Plot(show_label=False)
            gr.Examples(USER_EXAMPLES, inputs=ubox, label="Example users (the last one is a copy-paste bot)")
            ugo.click(tab_user, ubox, [uout, uplot])
        with gr.Tab("3. Grade it yourself"):
            gr.Markdown("Random tweets from the **held-out test set**: hand-labeled by me, never used in training. "
                        "Click a few times and watch the tally.\n\n" + scoreboard_md())
            tally = gr.State({})
            ggo = gr.Button("Draw a random held-out tweet", variant="primary")
            gout = gr.HTML()
            ggo.click(tab_grade, tally, [gout, tally])
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", help="score one tweet in the terminal and exit")
    ap.add_argument("--share", action="store_true", help="also make a temporary public gradio link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    if a.cli:
        for m, r in all_marks(a.cli).items():
            print(f"{m:7s} {r['side']:8s} {r['why']}")
    else:
        import gradio as gr
        build_ui().launch(share=a.share, server_port=a.port, css=CSS, theme=gr.themes.Soft())
