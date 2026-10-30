# stageC_users py who
import os, sys, json
import numpy as np, pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0
W = os.path.dirname(os.path.abspath(__file__))
TOP = 100

df = pd.read_pickle(os.path.join(W, "cleaned_flags.pkl"))
lut = pd.read_pickle(os.path.join(W, "stance_lut.pkl"))
df = df.merge(lut, on="cleaned_text", how="left")
print("rows", len(df), "missing stance", df.stance.isna().sum())

views = {
    "Mark 4": df,
    "Mark 5": df[df.long_enough & ~df.is_dup],
    "Mark 6": df[df.long_enough & ~df.is_dup & df.is_english],
}

# per user facts
raw_g = df.groupby("user_id")
dup_share = 1 - raw_g.cleaned_text.nunique() / raw_g.size()


def lang_share(texts):
    # share of a
    out = []
    for t in pd.unique(texts):
        try:
            out.append(detect(t) != "en")
        except LangDetectException:
            out.append(True)
    return float(np.mean(out))


res, examples = {}, {}
for name, v in views.items():
    g = v.groupby("user_id").stance.agg(["mean", "std", "count"])
    g = g[g["count"] >= 2]  # std needs 2
    top = g.sort_values("std", kind="stable").head(TOP)
    users = top.index
    nonen = pd.Series({u: lang_share(df.loc[df.user_id == u, "cleaned_text"]) for u in users})
    res[name] = dict(
        users_ranked=int(len(g)),
        tweets=int(len(v)),
        zero_std_users=int((g["std"] < 1e-3).sum()),
        top_copy_paste=float((dup_share[users] >= 0.5).mean()),
        top_non_english=float((nonen >= 0.5).mean()),
        top_std_median=float(top["std"].median()),
        top_count_median=float(top["count"].median()),
        top_abs_stance_median=float(top["mean"].abs().median()),
    )
    # a handful of
    ex = []
    for u in users[:8]:
        rows = v[v.user_id == u]
        ex.append(dict(n=int(len(rows)), distinct=int(rows.cleaned_text.nunique()),
                       std=float(top.loc[u, "std"]), text=rows.cleaned_text.iloc[0][:140]))
    examples[name] = ex
    print(name, json.dumps(res[name]))

json.dump(dict(results=res, examples=examples), open(os.path.join(W, "users_eval.json"), "w"), indent=1)
