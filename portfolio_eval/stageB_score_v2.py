# stageB_score_v2 py same
import sys, os, time, numpy as np, pandas as pd, torch

V4, work = sys.argv[1], sys.argv[2]
sys.path.insert(0, V4); import model as M

# only the unique
texts = pd.read_pickle(os.path.join(work, "cleaned_flags.pkl"))["cleaned_text"].unique().tolist()
print("unique cleaned texts:", len(texts), flush=True)

dev = torch.device("cuda")
m = M.BERTweetStanceClassifier()
m.load_state_dict(torch.load(os.path.join(work, "mark4_seed1.pt"), map_location="cpu"))
m = m.to(dev).eval().half()
tok = M.load_tokenizer()
pad = tok.pad_token_id

stance = np.zeros(len(texts), dtype=np.float32)
CHUNK, B = 4096, 256
t0 = time.time()
with torch.no_grad():
    for c0 in range(0, len(texts), CHUNK):
        ids = tok(texts[c0:c0 + CHUNK], max_length=M.MAX_LENGTH, truncation=True)["input_ids"]
        order = np.argsort([len(x) for x in ids])  # similar lengths together
        for k in range(0, len(order), B):
            idx = order[k:k + B]
            L = max(len(ids[i]) for i in idx)
            x = torch.full((len(idx), L), pad, dtype=torch.long)
            for j, i in enumerate(idx):
                x[j, :len(ids[i])] = torch.tensor(ids[i])
            a = (x != pad).long()
            stance[c0 + idx] = m(x.to(dev), a.to(dev))["stance"].float().cpu().numpy()
        del ids
        if (c0 // CHUNK) % 25 == 0:
            done = c0 + CHUNK
            print(f"{min(done, len(texts)):,}/{len(texts):,}  {time.time()-t0:.0f}s  "
                  f"eta {(time.time()-t0)/done*(len(texts)-done)/60:.0f} min", flush=True)

pd.DataFrame({"cleaned_text": texts, "stance": stance}).to_pickle(os.path.join(work, "stance_lut.pkl"))
print("done %.0fs" % (time.time() - t0), flush=True)
