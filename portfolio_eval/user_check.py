# do user scores
import json, os, sys
import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")
import model as fm
from stance_tags import split_tags, model_text

full = pd.read_pickle("outputs/cache/stage0_cleaned.pkl")
full["user_id"] = pd.to_numeric(full.user_id, errors="coerce")  # csv has floats
full = full[full.user_id.notna()]
full["side"] = full.cleaned_text.astype(str).map(lambda t: split_tags(t)[0])
tagged = full[full.side.notna()].groupby("user_id").side.agg(["count", "mean"])
sure = tagged[(tagged["count"] >= 3) & ((tagged["mean"] >= .8) | (tagged["mean"] <= .2))]
sure = pd.DataFrame({"truth": (sure["mean"] >= .8).astype(int)})
print("users with a clear declared side:", len(sure), sure.truth.value_counts().to_dict())

res = {}
for name, path in [("old 2nd run", "outputs/user_polarization_scores_2nd_run.csv"), ("Mark 7", "outputs/mark7/user_polarization_scores.csv")]:
    u = pd.read_csv(path).merge(sure, left_on="user_id", right_index=True)
    a = float(((u.mean_stance > 0).astype(int) == u.truth).mean())
    t, b = u[u.truth == 1], u[u.truth == 0]
    res[name] = dict(n=len(u), acc=a, trump=float((t.mean_stance > 0).mean()), biden=float((b.mean_stance < 0).mean()))
    print(f"{name} (tags still in the text): n={len(u)} agree {a:.1%}  Trump side {res[name]['trump']:.0%}  Biden side {res[name]['biden']:.0%}")

# the fair one
net = fm.BERTweetStanceClassifier()
net.load_state_dict(torch.load("model/mark7/best_model.pt", map_location="cpu"))
dev = torch.device("cuda")
net = net.to(dev).eval(); tok = fm.load_tokenizer()
rest = full[full.side.isna() & full.user_id.isin(sure.index)].copy()
n = rest.groupby("user_id").cleaned_text.transform("size")
rest = rest[n >= 2]
keep = sure.loc[sure.index.isin(rest.user_id)]
keep = pd.concat([g.sample(min(1000, len(g)), random_state=0) for _, g in keep.groupby("truth")])
rest = rest[rest.user_id.isin(keep.index)]
tx = rest.cleaned_text.astype(str).map(model_text).tolist()
out = []
with torch.no_grad():
    for i in range(0, len(tx), 64):
        e = fm.tokenize_batch(tx[i:i + 64], tok)
        out.append(net(e["input_ids"].to(dev), e["attention_mask"].to(dev))["probs"].float().cpu().numpy())
p = np.vstack(out)
rest["s"] = (p[:, 1] - p[:, 0]) / (p[:, 1] + p[:, 0])
u = rest.groupby("user_id").s.mean().to_frame().join(keep)
a = float(((u.s > 0).astype(int) == u.truth).mean())
tr, bi = float((u[u.truth == 1].s > 0).mean()), float((u[u.truth == 0].s < 0).mean())
res["Mark 7 tag-free"] = dict(n=len(u), tweets=len(rest), acc=a, trump=tr, biden=bi)
print(f"Mark 7 on tag-free tweets only: {len(u)} users, {len(rest)} tweets, agree {a:.1%}  Trump side {tr:.0%}  Biden side {bi:.0%}")
json.dump(res, open("portfolio_eval/user_check.json", "w"), indent=1)

# spam check on
h = pd.read_csv("data/heldout_test.csv")
tx = h.cleaned_text.astype(str).map(model_text).tolist()
out = []
with torch.no_grad():
    for i in range(0, len(tx), 64):
        e = fm.tokenize_batch(tx[i:i + 64], tok)
        out.append(net(e["input_ids"].to(dev), e["attention_mask"].to(dev))["probs"].float().cpu().numpy())
p = np.vstack(out); pred = p.argmax(1); sp = h.label.values == 2
print(f"spam: caught {(pred[sp] == 2).sum()} of {sp.sum()}, false alarms {(pred[~sp] == 2).sum()} of {(~sp).sum()}")
