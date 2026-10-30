# score every unique
import sys, os, time, numpy as np, pandas as pd, torch
V4, V6, work = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, V6); import clean as C
sys.path.insert(0, V4); import model as M
dev = torch.device("cuda")
m = M.BERTweetStanceClassifier().to(dev)
m.load_state_dict(torch.load(os.path.join(work, "mark4_seed1.pt"), map_location=dev)); m.eval().half()
tok = M.load_tokenizer()
raw = pd.read_pickle(os.path.join(work, "raw.pkl")).dropna(subset=["tweet", "user_id"])
texts = pd.Series(raw["tweet"].apply(C.clean_text).unique())
print("unique cleaned texts:", len(texts), flush=True)
t0 = time.time()
ids = tok(texts.tolist(), max_length=M.MAX_LENGTH, truncation=True)["input_ids"]
print("tokenized in %.0fs" % (time.time() - t0), flush=True)
order = np.argsort([len(x) for x in ids]); stance = np.zeros(len(ids), dtype=np.float32)
pad = tok.pad_token_id; B = 256
with torch.no_grad():
    for k in range(0, len(order), B):
        idx = order[k:k+B]; L = max(len(ids[i]) for i in idx)
        x = torch.full((len(idx), L), pad, dtype=torch.long); a = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            x[j, :len(ids[i])] = torch.tensor(ids[i]); a[j, :len(ids[i])] = 1
        stance[idx] = m(x.to(dev), a.to(dev))["stance"].float().cpu().numpy()
        if k % (B * 400) == 0: print(k, "%.0fs" % (time.time() - t0), flush=True)
pd.DataFrame({"cleaned_text": texts, "stance": stance}).to_pickle(os.path.join(work, "stance_lut.pkl"))
print("done %.0fs" % (time.time() - t0))
