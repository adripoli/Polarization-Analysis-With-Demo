# fig 7 fake
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figs as F  # reuse the palette

W = os.path.dirname(os.path.abspath(__file__))
U = json.load(open(os.path.join(W, "users_light.json")))
marks = ["Mark 4", "Mark 5", "Mark 6"]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
x = np.arange(3)
for ax, key, share, col, title in [
        (a, "all_identical_users", "all_identical_share", F.ORANGE, "Users whose every tweet is identical"),
        (b, "mostly_non_english_users", "mostly_non_english_share", F.VIOLET, "Users posting mostly non-English")]:
    v = [U[m][key] for m in marks]
    known = [vi is not None for vi in v]
    vals = [vi or 0 for vi in v]
    ax.bar(x, vals, 0.55, color=col, edgecolor=F.SURF, linewidth=2)
    top = max(vals)
    for xi, m, vi, k in zip(x, marks, vals, known):
        if not k:
            # mark 4 never
            ax.text(xi, top * 0.04, "not checked\nin Mark 4", ha="center", fontsize=8.5, color=F.INK2)
            continue
        lab = f"{vi:,}" + (f"\n{U[m][share]*100:.0f}% of users" if vi else "")
        ax.text(xi, vi + top * 0.03, lab, ha="center", va="bottom", fontsize=9, color=F.INK)
    ax.set_xticks(x); ax.set_xticklabels(marks); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, top * 1.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.set_title(title, loc="left")
fig.tight_layout(w_pad=3)
F.save(fig, "fig7_users.png")
print("ok")
