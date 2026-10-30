# rebuild the exact
import pickle, sys, pandas as pd
from sklearn.model_selection import train_test_split
P = r"C:\Users\Asus Gaming\Desktop\political_polarization"
OUT = sys.argv[1]
full = pickle.load(open(P + r"\outputs\cache\stage0_cleaned.pkl", "rb"))
print("cache:", full.shape, full.columns.tolist(), full["source"].value_counts().to_dict())
weak = full.copy()
weak["label"] = weak["source"].map({"biden": 0, "trump": 1}).astype(int)
weak["is_manual"] = False
man = pd.read_csv(P + r"\data\manual_labels.csv")
man["orig_label"] = man["label"]
man["label"] = man["label"].astype(int).replace({-1: 0})
man["is_manual"] = True
comb = pd.concat([weak[["cleaned_text","label","is_manual"]], man[["cleaned_text","label","is_manual","orig_label"]]], ignore_index=True)
comb = comb.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
print("combined:", len(comb), "manual kept:", comb.is_manual.sum())
tr, va = train_test_split(comb, train_size=0.8, stratify=comb["label"], random_state=42)
print("train", len(tr), "val", len(va))
mv = va[va.is_manual][["cleaned_text","orig_label"]].rename(columns={"orig_label":"label"})
mt = tr[tr.is_manual][["cleaned_text","orig_label"]].rename(columns={"orig_label":"label"})
mv["label"]=mv.label.astype(int); mt["label"]=mt.label.astype(int)
print("manual in val (TEST):", len(mv), mv.label.value_counts().to_dict())
print("manual in train:", len(mt), mt.label.value_counts().to_dict())
mv.to_csv(OUT + "/test_heldout.csv", index=False)
mt.to_csv(OUT + "/train_manual.csv", index=False)
# also the full
m_all = man.drop_duplicates("cleaned_text")[["cleaned_text","orig_label"]].rename(columns={"orig_label":"label"})
m_all.to_csv(OUT + "/manual_all_dedup.csv", index=False)
print("manual all dedup:", len(m_all), m_all.label.value_counts().to_dict())
# conflicting duplicate labels
g = man.groupby("cleaned_text")["orig_label"].nunique()
print("texts with conflicting labels:", (g>1).sum())
