# grade Mark 4
import json, os, sys
import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "demo"), ROOT]
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")
import app
import model as fm
from stance_tags import model_text

m7 = fm.BERTweetStanceClassifier()
m7.load_state_dict(torch.load("model/mark7/best_model.pt", map_location="cpu"))
m7 = m7.to(app._device()).eval()
tok = fm.load_tokenizer()


def mark7_trump(texts):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = fm.tokenize_batch([model_text(t) or "." for t in texts[i:i + 32]], tok)
            p = m7(enc["input_ids"].to(app._device()), enc["attention_mask"].to(app._device()))["probs"].float().cpu().numpy()
            out.append(p)
    p = np.vstack(out)
    return p[:, 1] / (p[:, 0] + p[:, 1]), p[:, 2]


def load(path):
    d = pd.read_csv(path)
    d = d[d.label != 2].copy()
    d["y"] = (d.label == 1).astype(int)  # hand files use
    return d


tests = {"heldout": load("data/heldout_test.csv"), "shortcut": load("data/shortcut_test.csv"), "fresh": load("data/fresh_probe.csv")}
res = {}
for name, d in tests.items():
    texts = d.cleaned_text.astype(str).tolist()
    scores = {"Mark 4": app.model_probs("Mark 4", texts)[:, 1], "Old final": app.model_probs("Final", texts)[:, 1]}
    scores["Mark 7"], spam = mark7_trump(texts)
    res[name] = {"n": len(d)}
    print(f"\n== {name} (n={len(d)}, Trump {d.y.sum()}, Biden {(1 - d.y).sum()})")
    for m, s in scores.items():
        pred = (s > 0.5).astype(int)
        tr, bi = (pred[d.y == 1] == 1).mean(), (pred[d.y == 0] == 0).mean()
        res[name][m] = dict(acc=float((pred == d.y).mean()), trump=float(tr), biden=float(bi), distinct=int(len(np.unique(s.round(2)))))
        print(f"  {m:10s} acc {res[name][m]['acc']:.1%}  Trump side {tr:.0%}  Biden side {bi:.0%}  distinct scores {res[name][m]['distinct']}")
    print(f"  Mark 7 mean P(spam) {spam.mean():.3f}")
    if name == "fresh":
        for t, y, a, b in zip(texts, d.y, scores["Mark 4"], scores["Mark 7"]):
            print(f"   {'T' if y else 'B'}  M4 {a:.2f}  M7 {b:.2f}  {'ok ' if (b > .5) == y else 'BAD'} {t}")

json.dump(res, open("portfolio_eval/mark7_eval.json", "w"), indent=1)
