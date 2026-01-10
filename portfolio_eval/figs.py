# figs py every
import json, glob, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(W, "..", "figs")
os.makedirs(F, exist_ok=True)

# palette reference instance
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"
GRAY = "#c9c8c2"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left",
})


def save(fig, name):
    fig.savefig(os.path.join(F, name), dpi=220, bbox_inches="tight")
    plt.close(fig)


kw = {c: json.load(open(os.path.join(W, f"kw_{c}.json"))) for c in
      "0712dd3 1ba92a1 112781e 5fd63dd 05dec46 22f45be 971c880 0c80a29 6970d94 a37cba4 a2f78b5".split()}
m4 = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(W, "m4_full_*.json")))]
m4f = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(W, "m4_frozen_*.json")))]
fin = json.load(open(os.path.join(W, "final_eval.json")))
test = pd.read_csv(os.path.join(W, "test_heldout.csv"))
n_pol = int((test.label != 2).sum())


# fig 1 how
def fig_pipeline():
    marks = [
        ("Mark 1", ["Tweets", "Keyword check", "User scores"], {"Keyword check"}),
        ("Mark 2", ["Tweets", "Keyword check", "Adjective flip", "User scores"], {"Adjective flip"}),
        ("Mark 3", ["Tweets", "Keyword check", "Adjective flip", "Distance\nweighting", "User scores"],
         {"Distance\nweighting"}),
        ("Mark 4", ["Tweets", "Tokenize", "BERTweet", "User scores"], {"Tokenize", "BERTweet"}),
        ("Mark 5", ["Tweets", "Clean\n(>5 words, dedupe)", "Tokenize", "BERTweet", "User scores"],
         {"Clean\n(>5 words, dedupe)"}),
        ("Mark 6", ["Tweets", "Clean", "English check", "Tokenize", "BERTweet", "User scores"], {"English check"}),
        ("Final", ["Tweets", "Clean +\nEnglish", "Tokenize", "BERTweet\n(Biden/Trump/\nSpam)",
                   "Spam filter +\npolarization\nscore", "GMM"],
         {"BERTweet\n(Biden/Trump/\nSpam)", "Spam filter +\npolarization\nscore", "GMM"}),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 11); ax.set_ylim(-0.6, len(marks)); ax.axis("off")
    bw, gap = 1.42, 0.22
    for r, (name, stages, new) in enumerate(marks):
        y = len(marks) - 1 - r
        era = "ML" if r >= 3 else "rules"
        ax.text(0.05, y + 0.02, name, fontsize=11, fontweight="bold", va="center", color=INK)
        x = 0.95
        for i, s in enumerate(stages):
            isnew = s in new
            fc = (BLUE if era == "ML" else ORANGE) if isnew else "#efeeea"
            tc = "white" if isnew else INK
            box = FancyBboxPatch((x, y - 0.3), bw, 0.6, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 fc=fc, ec="none")
            ax.add_patch(box)
            ax.text(x + bw / 2, y, s, ha="center", va="center", fontsize=7.4, color=tc, linespacing=1.05,
                    fontweight="bold" if isnew else "normal")
            if i < len(stages) - 1:
                ax.annotate("", xy=(x + bw + gap - 0.02, y), xytext=(x + bw + 0.02, y),
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, mutation_scale=8))
            x += bw + gap
    ax.plot([0, 11], [3.5, 3.5], color=GRID, lw=1, ls="--")
    ax.text(10.95, 3.56, "hand-written rules ↑", ha="right", va="bottom", fontsize=8.5, color=INK2, style="italic")
    ax.text(10.95, 3.44, "machine learning ↓", ha="right", va="top", fontsize=8.5, color=INK2, style="italic")
    ax.text(0.95, -0.55, "Colored box = the stage that Mark added.  Orange = rule-based era, blue = ML era.",
            fontsize=8.5, color=INK2)
    ax.set_title("How the pipeline evolved, Mark 1 to final", loc="left", pad=6)
    save(fig, "fig1_pipeline.png")


# fig 2 correct
def fig_accuracy():
    rows = []
    for label, c in [("Mark 1  keywords", "1ba92a1"), ("Mark 2  + adjective flip", "22f45be"),
                     ("Mark 3  + distance weighting", "a2f78b5")]:
        t = kw[c]["test_heldout"]
        rows.append((label, t["accuracy"], t["wrong_side"], 1 - t["coverage"], None))
    fz = np.array([d["pol_acc"] for d in m4f])
    rows.append(("Mark 4a  BERTweet, frozen encoder", fz.mean(), 1 - fz.mean(), 0, fz.std()))
    a = np.array([d["pol_acc"] for d in m4])
    rows.append(("Mark 4  BERTweet, fine-tuned", a.mean(), 1 - a.mean(), 0, a.std()))
    f = fin["test_heldout"]["pol_acc"]
    rows.append(("Old final model (stance head)", f, 1 - f, 0, None))
    e7 = json.load(open(r"C:\Users\Asus Gaming\Desktop\political_polarization\portfolio_eval\mark7_eval.json"))
    s7 = json.load(open(r"C:\Users\Asus Gaming\Desktop\political_polarization\portfolio_eval\mark7_seeds.json"))
    a7 = np.array([e7["heldout"]["Mark 7"]["acc"]] + [v["heldout"] for v in s7.values()])
    rows.append(("Mark 7  shortcut fixed", a7.mean(), 1 - a7.mean(), 0, a7.std()))
    fig, ax = plt.subplots(figsize=(10, 4.7))
    ax.grid(axis="y", visible=False)
    for i, (lab, c, w, n, sd) in enumerate(rows[::-1]):
        ax.barh(i, c, color=BLUE, height=0.62, edgecolor=SURF, linewidth=2)
        ax.barh(i, w, left=c, color=ORANGE, height=0.62, edgecolor=SURF, linewidth=2)
        ax.barh(i, n, left=c + w, color=GRAY, height=0.62, edgecolor=SURF, linewidth=2)
        txt = f"{c*100:.0f}%" + (f" ± {sd*100:.1f}" if sd is not None else "")
        if c < 0.15:
            ax.text(c + w + 0.012, i, txt + " correct", va="center", ha="left", fontsize=9,
                    fontweight="bold", color=INK)
        else:
            ax.text(c - 0.012, i, txt, va="center", ha="right", fontsize=9, fontweight="bold", color="white")
        if n > 0.3:
            ax.text(c + w + n - 0.012, i, f"no call {n*100:.0f}%", va="center", ha="right", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.axvline(82 / n_pol, color=INK2, lw=1, ls=":")
    ax.text(82 / n_pol + 0.006, -0.62, "always guessing 'Trump' = 53%", fontsize=8, color=INK2, va="center")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="correct"), Patch(color=ORANGE, label="wrong side"),
                       Patch(color=GRAY, label="no call (scored 0)")],
              loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=3, frameon=False)
    ax.set_ylim(-0.85, len(rows) - 0.5)
    ax.set_title(f"Accuracy on the same {n_pol} held-out hand-labeled political tweets", loc="left")
    save(fig, "fig2_accuracy.png")


# fig 3 every
def fig_commits():
    order = [("0712dd3", "first keyword\nscoring"), ("1ba92a1", "Mark 1\nfinal"),
             ("112781e", "negative-adj\nflip"), ("5fd63dd", "bulk keywords"),
             ("05dec46", "positive\nadjectives"), ("22f45be", "Mark 2\nfinal"),
             ("971c880", "distance\nweight v1"), ("0c80a29", "div/0 fix"), ("6970d94", "distance cap"),
             ("a37cba4", "vote per\nkeyword"), ("a2f78b5", "Mark 3\nfinal")]
    acc = [kw[c]["manual_all_dedup"]["accuracy"] for c, _ in order]
    cov = [kw[c]["manual_all_dedup"]["coverage"] for c, _ in order]
    prec = [kw[c]["manual_all_dedup"]["precision_called"] for c, _ in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for y, col, lab in [(prec, VIOLET, "right when it made a call"), (cov, AQUA, "made a call at all (coverage)"),
                        (acc, BLUE, "overall accuracy")]:
        ax.plot(x, y, color=col, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=lab)
        ax.text(x[-1] + 0.15, y[-1], f"{y[-1]*100:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n{l}" for c, l in order], fontsize=7.3)
    ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(-0.4, len(order) - 0.2)
    for b in (1.5, 5.5):
        ax.axvline(b, color=GRID, lw=1, ls="--")
    for xm, t in [(0.5, "Mark 1"), (3.5, "Mark 2"), (8.5, "Mark 3")]:
        ax.text(xm, 0.96, t, ha="center", fontsize=9.5, fontweight="bold", color=INK2)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), frameon=False, fontsize=9)
    n = kw["a2f78b5"]["manual_all_dedup"]["n"]
    ax.set_title(f"Every rule-based commit, rerun on all {n} hand-labeled political tweets", loc="left")
    save(fig, "fig3_commits.png")


# fig 4 frozen
def fig_loss():
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for runs, col, lab in [(m4f, ORANGE, "Mark 4a: frozen encoder, train head only (bd4d639)"),
                           (m4, BLUE, "Mark 4: fine-tune everything (e554bad)")]:
        for j, d in enumerate(runs):
            L = pd.Series(d["losses"]).rolling(8, min_periods=1).mean().values
            ep = np.linspace(0, 1, len(L))
            ax.plot(ep, L, color=col, lw=2 if j == 0 else 1.1, alpha=1 if j == 0 else 0.35,
                    label=lab if j == 0 else None)
    ax.axhline(np.log(2), color=INK2, lw=1, ls=":")
    ax.text(0.995, np.log(2) + 0.012, "ln 2 = coin-flip loss", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("fraction of training run"); ax.set_ylabel("training loss (8-step rolling mean)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("Rerun: the frozen version learned too, just less (70% vs 76%)", loc="left")
    save(fig, "fig4_loss.png")


# fig 5 every
def fig_timeline():
    rows = [l.strip().split("|", 2) for l in open(os.path.join(W, "commits.txt"), encoding="utf8") if l.strip()]
    starts = [("da08515", "Mark 1"), ("112781e", "Mark 2"), ("9f440ef", "Mark 3"), ("5343537", "Mark 4"),
              ("1503a65", "Mark 5"), ("4b58e61", "Mark 6"), ("f5a0e74", "Final refactor"), ("8ce8559", "Publish")]
    lanes = [s[1] for s in starts]
    cur, data = None, []
    for h, d, msg in rows:
        for sh, nm in starts:
            if h == sh:
                cur = nm
        data.append((pd.Timestamp(d).tz_convert("UTC").tz_localize(None), cur, h, msg))
    t = pd.DataFrame(data, columns=["date", "mark", "hash", "msg"])
    t = t[t.mark != "Publish"]
    lanes = lanes[:-1]
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.grid(axis="y", visible=False)
    for i, ln in enumerate(lanes):
        s = t[t.mark == ln]
        col = ORANGE if i < 3 else BLUE if i < 6 else AQUA
        # jitter same day
        off = s.groupby(s.date.dt.date).cumcount() * pd.Timedelta(hours=6.5)
        ax.scatter(s.date + off, [len(lanes) - 1 - i] * len(s), s=30, color=col, edgecolor=SURF, linewidth=0.8, zorder=3)
        ax.text(s.date.max() + off.max() + pd.Timedelta(days=1.2), len(lanes) - 1 - i,
                f"{len(s)} commit{'s' if len(s) > 1 else ''}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(lanes))); ax.set_yticklabels(lanes[::-1])
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    ax.set_xlim(t.date.min() - pd.Timedelta(days=2), t.date.max() + pd.Timedelta(days=9))
    ax.set_title(f"{len(t)} commits, Oct 10 to Dec 15, 2025 (rules orange, ML blue, cleanup green)", loc="left")
    save(fig, "fig5_timeline.png")


# fig 6 final
def fig_final():
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm
    u = pd.read_csv(r"C:\Users\Asus Gaming\Desktop\political_polarization\outputs\user_polarization_scores_2nd_run.csv")
    X = u.mean_stance.values.reshape(-1, 1)
    g = GaussianMixture(2, random_state=42, max_iter=200).fit(X)
    o = np.argsort(g.means_.ravel()); m = g.means_.ravel()[o]; sd = np.sqrt(g.covariances_.ravel())[o]; w = g.weights_[o]
    x = np.linspace(-1, 1, 600)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a.hist(u.mean_stance, bins=80, range=(-1, 1), density=True, color="#dcdbd5", edgecolor=SURF, linewidth=0.4)
    p0, p1 = w[0] * norm.pdf(x, m[0], sd[0]), w[1] * norm.pdf(x, m[1], sd[1])
    a.fill_between(x, np.minimum(p0, p1), color=INK2, alpha=0.25, lw=0)
    a.plot(x, p0, color=BLUE, lw=2); a.plot(x, p1, color=RED, lw=2)
    a.text(m[0], p0.max() + 0.04, f"pro-Biden cluster\nmean {m[0]:.2f}, {w[0]*100:.0f}% of users", ha="center", fontsize=8.5, color=INK)
    a.text(m[1], p1.max() + 0.04, f"pro-Trump cluster\nmean {m[1]:.2f}, {w[1]*100:.0f}% of users", ha="center", fontsize=8.5, color=INK)
    ov = np.trapezoid(np.minimum(p0, p1), x)
    a.annotate(f"overlap = {ov:.3f}", xy=(-0.05, 0.12), xytext=(0.4, 0.06), fontsize=8.5, color=INK2,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a.set_ylim(0, 1.15); a.set_xlabel("user mean stance  (-1 = pro-Biden, +1 = pro-Trump)"); a.set_ylabel("density")
    a.set_title(f"{len(u):,} users after cleaning + spam filtering", loc="left")
    # right does posting
    bins = [3, 4, 6, 10, 20, 50, 200, 100000]
    u["bin"] = pd.cut(u.tweet_count, bins, right=False)
    med = u.groupby("bin", observed=True)[["polarization_raw", "polarization_weighted"]].median()
    xs = np.arange(len(med))
    b.plot(xs, med.polarization_raw, color=VIOLET, lw=2, marker="o", ms=6, mec=SURF, label="raw score (no tweet-count weight)")
    b.plot(xs, med.polarization_weighted, color=AQUA, lw=2, marker="o", ms=6, mec=SURF, label="final score (x log tweet count)")
    b.set_xticks(xs); b.set_xticklabels(["3", "4-5", "6-9", "10-19", "20-49", "50-199", "200+"][:len(med)], fontsize=8.5)
    b.set_xlabel("tweets per user"); b.set_ylabel("median polarization score"); b.set_ylim(0, 1)
    b.legend(frameon=False, fontsize=8, loc="upper left")
    b.set_title("The activity trend comes from the weight", loc="left")
    fig.tight_layout(w_pad=3)
    save(fig, "fig6_final.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["pipeline", "accuracy", "commits", "loss"]
    for w in which:
        globals()["fig_" + w]()
    print("ok", which)
