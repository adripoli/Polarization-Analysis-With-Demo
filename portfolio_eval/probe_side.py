# probe_side py does
import sys, os, re, json
import numpy as np, pandas as pd, torch
P = r"C:\Users\Asus Gaming\Desktop\political_polarization"
sys.path.insert(0, P); os.chdir(P)
from model import BERTweetStanceClassifier, load_tokenizer, tokenize_batch
from stage0_cleaning import clean_text

dev = torch.device("cuda")
m = BERTweetStanceClassifier(); m.load_state_dict(torch.load(r"model\bertweet_finetuned\best_model.pt", map_location="cpu"))
m = m.to(dev).eval(); tok = load_tokenizer()

def stance(texts):
    with torch.no_grad():
        e = tokenize_batch(texts, tok)
        p = m(e["input_ids"].to(dev), e["attention_mask"].to(dev))["probs"].float().cpu().numpy()
    return p[:, 1] / (p[:, 0] + p[:, 1]) * 2 - 1   # 1 trump 1

pro_t = ["Four more years! #MAGA", "The economy was booming before covid, give him another term",
         "Sleepy Joe and the radical left want to defund the police", "Trump 2020! Keep America Great #KAG",
         "Biden is a corrupt liar, vote Trump", "Best president of my lifetime #Trump2020"]
pro_b = ["Vote him out #BidenHarris2020", "Trump lied about covid and people died",
         "Can't wait for a president who believes in science", "Early voted for Biden today!",
         "Trump is the worst president in history", "Joe Biden will restore decency #VoteBlue"]
handfmt = lambda t: re.sub(r"\s+", " ", t.replace("#", "")).strip()
for name, f in [("stage0 clean (demo)", clean_text), ("hand-label style", handfmt)]:
    st = stance([f(t) for t in pro_t]); sb = stance([f(t) for t in pro_b])
    print(f"\n== {name}: pro-Trump right {np.mean(st > 0):.0%}  pro-Biden right {np.mean(sb < 0):.0%}")
    for t, v in zip(pro_t, st): print(f"  T {v:+.2f}  {t}")
    for t, v in zip(pro_b, sb): print(f"  B {v:+.2f}  {t}")

# held out by
d = pd.read_csv(os.path.join(sys.argv[1], "test_heldout.csv")); d = d[d.label != 2]
s = stance(d.cleaned_text.tolist()[:64]) if False else np.concatenate([stance(d.cleaned_text.tolist()[i:i+32]) for i in range(0, len(d), 32)])
d["pred"] = np.where(s > 0, 1, -1)
print("\nheld-out accuracy by true side:", d.groupby("label").apply(lambda g: (g.pred == g.label).mean()).round(3).to_dict())
print("share predicted Trump:", (d.pred == 1).mean().round(3), " true share Trump:", (d.label == 1).mean().round(3))
