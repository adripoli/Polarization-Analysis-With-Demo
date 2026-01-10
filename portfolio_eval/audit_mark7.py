# is Mark 7
import os, sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")
from stage3_training import assign_weak_labels, locked_keys
from stance_tags import PRO_BIDEN, PRO_TRUMP, model_text, text_key
from config import LOCKED_TEST_CSVS, MANUAL_LABELS_CSV

full = pd.read_pickle("outputs/cache/stage0_cleaned.pkl")
locked = locked_keys()
weak = assign_weak_labels(full, locked)
hand = pd.read_csv(MANUAL_LABELS_CSV)
hand = hand[~hand.cleaned_text.astype(str).map(text_key).isin(locked)]
train = pd.concat([weak.cleaned_text, hand.cleaned_text.astype(str).map(model_text)], ignore_index=True)
keys = set(train.map(text_key))

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
X = vec.fit_transform(train)
for path in LOCKED_TEST_CSVS:
    d = pd.read_csv(path)
    d = d[d.label != 2].reset_index(drop=True)
    texts = d.cleaned_text.astype(str).map(model_text)
    exact = texts.map(text_key).isin(keys).sum()
    best = np.concatenate([(vec.transform(texts[i:i + 64]) @ X.T).max(axis=1).toarray().ravel() for i in range(0, len(d), 64)])
    d["nearest"] = best
    d.to_csv(path.replace("data/", "portfolio_eval/near_").replace(".csv", ".csv"), index=False)
    print(f"{path}: n={len(d)} exact leaks={exact}  nearest>=0.8: {(best >= .8).sum()}  >=0.6: {(best >= .6).sum()}  median {np.median(best):.2f}")

# blind pool no
tags = full.cleaned_text.astype(str).str.findall(r"#(\w+)").apply(lambda x: {t.lower() for t in x})
clean = full[~tags.apply(lambda s: bool(s & (PRO_BIDEN | PRO_TRUMP)))].copy()
clean["t"] = clean.cleaned_text.astype(str).map(model_text)
clean = clean[clean.t.str.split().str.len().between(8, 45)].drop_duplicates("t")
clean = clean[~clean.t.map(text_key).isin(keys | locked)]
s = clean.sample(120, random_state=2026)[["t"]].reset_index(drop=True)
s.to_csv("portfolio_eval/blind_pool.csv", index_label="id")
print("\nuntagged pool:", len(clean), "of", len(full))
for i, t in s.t.items():
    print(f"{i}|{t[:230]}")
