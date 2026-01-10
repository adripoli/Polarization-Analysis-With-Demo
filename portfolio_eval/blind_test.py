# random tweets with
import json, os, sys
import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "demo"), ROOT]
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")
import app
import model as fm
from stance_tags import model_text

BIDEN = [0, 1, 2, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 22, 24, 30, 32, 36, 39, 42, 44, 46, 48, 51, 52, 54, 56, 57, 60,
         67, 68, 70, 72, 74, 79, 81, 83, 87, 92, 94, 100, 101, 102, 103, 106, 109, 113, 114, 117]
TRUMP = [6, 21, 27, 31, 33, 37, 40, 49, 53, 55, 73, 75, 78, 82, 84, 85, 86, 89, 91, 95, 96, 99, 104, 105, 108, 110, 111, 116]

p = pd.read_csv("portfolio_eval/blind_pool.csv", index_col="id")
d = pd.concat([p.loc[BIDEN].assign(y=0), p.loc[TRUMP].assign(y=1)])
d[["t", "y"]].to_csv("data/blind_test.csv", index=False)
texts = d.t.astype(str).tolist()

m7 = fm.BERTweetStanceClassifier()
m7.load_state_dict(torch.load("model/mark7/best_model.pt", map_location="cpu"))
m7 = m7.to(app._device()).eval()
tok = fm.load_tokenizer()
out = []
with torch.no_grad():
    for i in range(0, len(texts), 32):
        e = fm.tokenize_batch([model_text(t) for t in texts[i:i + 32]], tok)
        out.append(m7(e["input_ids"].to(app._device()), e["attention_mask"].to(app._device()))["probs"].float().cpu().numpy())
q = np.vstack(out)
scores = {"Mark 4": app.model_probs("Mark 4", texts)[:, 1], "Old final": app.model_probs("Final", texts)[:, 1], "Mark 7": q[:, 1] / (q[:, 0] + q[:, 1])}

res = {"n": len(d), "trump": int(d.y.sum()), "biden": int((1 - d.y).sum()), "always_biden": float((d.y == 0).mean())}
print(f"blind n={len(d)} Trump {d.y.sum()} Biden {(1 - d.y).sum()}  (always saying Biden gets {res['always_biden']:.0%})")
for m, s in scores.items():
    pr = (s > .5).astype(int)
    res[m] = dict(acc=float((pr == d.y).mean()), trump=float((pr[d.y == 1] == 1).mean()), biden=float((pr[d.y == 0] == 0).mean()))
    print(f"  {m:10s} acc {res[m]['acc']:.1%}  Trump side {res[m]['trump']:.0%}  Biden side {res[m]['biden']:.0%}")
for t, y, s in zip(texts, d.y, scores["Mark 7"]):
    if (s > .5) != y:
        print(f"  miss {'T' if y else 'B'} {s:.2f} {t[:130]}")

for path, name in [("heldout_test", "heldout"), ("shortcut_test", "shortcut"), ("fresh_probe", "fresh")]:
    n = pd.read_csv(f"portfolio_eval/near_{path}.csv")
    keep = n[n.nearest < .6]
    tx = keep.cleaned_text.astype(str).tolist()
    y = (keep.label == 1).astype(int).values
    o = []
    with torch.no_grad():
        for i in range(0, len(tx), 32):
            e = fm.tokenize_batch([model_text(t) or "." for t in tx[i:i + 32]], tok)
            o.append(m7(e["input_ids"].to(app._device()), e["attention_mask"].to(app._device()))["probs"].float().cpu().numpy())
    o = np.vstack(o); s = o[:, 1] / (o[:, 0] + o[:, 1])
    res[f"{name}_no_near_dupes"] = dict(n=len(keep), acc=float(((s > .5).astype(int) == y).mean()))
    print(f"{name} without near-dupes: n={len(keep)} acc {res[f'{name}_no_near_dupes']['acc']:.1%}")
json.dump(res, open("portfolio_eval/blind_eval.json", "w"), indent=1)
