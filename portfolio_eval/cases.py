import sys, importlib, json, re, pandas as pd
S = sys.argv[1]
cases = ["Those MAGA people are crazy",
         "trump is crazy good at rallies",
         "Incompetent. Just like everyone else involved in the trump clown car. VoteBiden",
         "MAGA! Trump 2020, four more years, the best president ever",
         "Ugh, of course the MAGA crowd thinks this is brilliant. Great job guys. VoteBiden"]
res = {}
for c in ["1ba92a1", "22f45be", "a2f78b5"]:
    for m in ["analyze_tweets", "keywords", "adjectives"]:
        sys.modules.pop(m, None)
    sys.path.insert(0, f"{S}/versions/{c}")
    at = importlib.import_module("analyze_tweets")
    res[c] = [at.score_tweet(t) for t in cases]
    sys.path.pop(0)
for i, t in enumerate(cases):
    print([res[c][i] for c in res], t)
# diagnostics with mark
sys.path.insert(0, f"{S}/versions/22f45be")
for m in ["keywords", "adjectives"]: sys.modules.pop(m, None)
import keywords as K, adjectives as A
d = pd.read_csv(f"{S}/work/manual_all_dedup.csv")
low = d.cleaned_text.str.lower()
haskw = low.apply(lambda t: any(k in t for k in K.TRUMP_KEYWORDS + K.BIDEN_KEYWORDS))
hasadj = low.apply(lambda t: any(a in t for a in A.NEGATIVE_ADJECTIVES + A.POSITIVE_ADJECTIVES))
print("labeled tweets w/ keyword:", haskw.mean(), " w/ keyword AND adjective:", (haskw & hasadj).mean(), (haskw & hasadj).sum())
# which keywords fire
raw = pd.read_pickle(f"{S}/work/raw.pkl").tweet.fillna("").str.lower()
only45 = raw.apply(lambda t: ("45" in t) and not any(k in t for k in K.TRUMP_KEYWORDS + K.BIDEN_KEYWORDS if k != "45"))
print("full-data tweets 'pro-trump' only because they contain '45':", only45.sum(), only45.mean())
magazine = raw.str.contains("magazine").sum(); print("tweets containing 'magazine' (substring hits 'maga'):", magazine)
