# stageC_light py model
import os, json
import pandas as pd

W = os.path.dirname(os.path.abspath(__file__))
df = pd.read_pickle(os.path.join(W, "cleaned_flags.pkl"))[["user_id", "cleaned_text", "long_enough", "is_dup", "is_english"]]

views = {
    "Mark 4": df,
    "Mark 5": df[df.long_enough & ~df.is_dup],
    "Mark 6": df[df.long_enough & ~df.is_dup & df.is_english],
}
res = {}
for name, v in views.items():
    g = v.groupby("user_id").agg(n=("cleaned_text", "size"), uniq=("cleaned_text", "nunique"),
                                 eng=("is_english", "mean"))
    g = g[g.n >= 2]  # need 2 tweets
    all_same = g.uniq == 1
    res[name] = dict(
        tweets=int(len(v)),
        users_2plus=int(len(g)),
        all_identical_users=int(all_same.sum()),
        all_identical_share=float(all_same.mean()),
        all_identical_tweets=int(g.loc[all_same, "n"].sum()),
        mostly_non_english_users=int((g.eng < 0.5).sum()) if name != "Mark 4" else None,
        mostly_non_english_share=float((g.eng < 0.5).mean()) if name != "Mark 4" else None,
    )
    print(name, json.dumps(res[name]))
# for mark 4
g4 = views["Mark 5"].groupby("user_id").agg(n=("cleaned_text", "size"))
json.dump(res, open(os.path.join(W, "users_light.json"), "w"), indent=1)
