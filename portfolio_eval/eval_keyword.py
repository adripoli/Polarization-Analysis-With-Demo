# run ONE old
import sys, os, json, importlib, pandas as pd
vdir, work = sys.argv[1], sys.argv[2]
sys.path.insert(0, vdir)
at = importlib.import_module("analyze_tweets")
out = {"commit": os.path.basename(vdir)}
for name in ["test_heldout", "manual_all_dedup"]:
    d = pd.read_csv(os.path.join(work, name + ".csv"))
    d["pred"] = d["cleaned_text"].apply(at.score_tweet)
    pol = d[d.label != 2]
    spam = d[d.label == 2]
    out[name] = dict(
        n=len(pol),
        accuracy=float((pol.pred == pol.label).mean()),
        coverage=float((pol.pred != 0).mean()),
        precision_called=float((pol.pred == pol.label)[pol.pred != 0].mean()) if (pol.pred != 0).any() else None,
        wrong_side=float(((pol.pred != 0) & (pol.pred != pol.label)).mean()),
        spam_called=float((spam.pred != 0).mean()),
    )
    if name == "test_heldout":
        d.to_csv(os.path.join(work, f"preds_{out['commit']}.csv"), index=False)
raw = pd.read_pickle(os.path.join(work, "raw.pkl"))
s = raw["tweet"].fillna("").apply(at.score_tweet)  # mark 3 crashes
out["full"] = dict(n=len(s), neutral=float((s == 0).mean()), trump=float((s == 1).mean()), biden=float((s == -1).mean()))
json.dump(out, open(os.path.join(work, f"kw_{out['commit']}.json"), "w"), indent=1)
print(json.dumps(out))
