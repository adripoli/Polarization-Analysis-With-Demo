# fig 7 who
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figs as F  # reuse the palette

W = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(os.path.join(W, "users_light.json")))
T = json.load(open(os.path.join(W, "users_eval.json")))["results"]
marks = ["Mark 4", "Mark 5", "Mark 6"]
x = np.arange(3)
fig, (a, b, c) = plt.subplots(1, 3, figsize=(11.5, 3.5), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))

# left the top
bw = 0.36
for k, (key, col, lab) in enumerate([("top_copy_paste", F.ORANGE, "copy-paste accounts"),
                                     ("top_non_english", F.VIOLET, "mostly non-English")]):
    v = [T[m][key] for m in marks]
    a.bar(x + (k - 0.5) * bw, v, bw, color=col, edgecolor=F.SURF, linewidth=2, label=lab)
    for xi, vi in zip(x, v):
        a.text(xi + (k - 0.5) * bw, vi + 0.02, f"{vi*100:.0f}%", ha="center", fontsize=8.5, color=F.INK)
a.set_ylim(0, 1.18); a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
a.legend(frameon=False, fontsize=8, loc="upper right")
a.set_title("Top 100 “most polarized” users", loc="left", fontsize=11)

# middle right population
for ax, key, col, title in [(b, "all_identical_users", F.ORANGE, "Users whose every\ntweet is identical"),
                            (c, "mostly_non_english_users", F.VIOLET, "Users posting\nmostly non-English")]:
    v = [L[m][key] for m in marks]
    vals = [vi or 0 for vi in v]; top = max(vals)
    ax.bar(x, vals, 0.55, color=col, edgecolor=F.SURF, linewidth=2)
    for xi, vi, raw in zip(x, vals, v):
        ax.text(xi, (vi + top * 0.03) if raw is not None else top * 0.04,
                f"{vi:,}" if raw is not None else "not\nchecked", ha="center", va="bottom", fontsize=8.5,
                color=F.INK if raw is not None else F.INK2)
    ax.set_ylim(0, top * 1.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y/1000:.0f}k" if y else "0"))
    ax.set_title(title, loc="left", fontsize=11)
for ax in (a, b, c):
    ax.set_xticks(x); ax.set_xticklabels(marks); ax.grid(axis="x", visible=False)
fig.tight_layout(w_pad=2.5)
F.save(fig, "fig7_users.png")
print("ok")
