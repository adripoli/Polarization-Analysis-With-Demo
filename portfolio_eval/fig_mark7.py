# fig_mark7 py mark
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = r"C:\Users\Asus Gaming\Desktop\political_polarization\portfolio_eval"
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
e, b = json.load(open(os.path.join(R, "mark7_eval.json"))), json.load(open(os.path.join(R, "blind_eval.json")))
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
COL = {"Mark 4": "#eb6834", "Old final": "#8a8984", "Mark 7": "#2a78d6"}
tests = [("Held-out\nyour hand labels", e["heldout"], 155), ("Supportive tweets\nname only their own side", e["shortcut"], 77),
         ("Fresh tweets\nwritten before training", e["fresh"], 40), ("Random untagged\nlabeled blind", b, 77)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, axes = plt.subplots(1, 4, figsize=(10, 3.3), sharey=True)
for ax, (name, d, n) in zip(axes, tests):
    for i, m in enumerate(COL):
        v = d[m]["acc"]
        ax.bar(i, v, color=COL[m], width=0.68, edgecolor=SURF, linewidth=2)
        ax.text(i, v + 0.02, f"{v*100:.0f}%", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_title(f"{name}\n(n={n})", fontsize=9.5, color=INK, loc="left")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Mark 4", "Old\nfinal", "Mark 7"], fontsize=8.5, color=INK2)
    ax.set_ylim(0, 1.1); ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0%", "50%", "100%"], color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
fig.tight_layout(w_pad=1.6)
fig.savefig(os.path.join(F, "fig8_mark7.png"), dpi=220, bbox_inches="tight")
print("ok")
