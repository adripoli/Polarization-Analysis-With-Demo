# retrain an old
import sys, os, json, importlib, random
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, random_split
vdir, work, mode, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, vdir)
M = importlib.import_module("model"); T = importlib.import_module("train")
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
dev = torch.device("cuda")
tok = M.load_tokenizer()
df = pd.read_csv(os.path.join(work, "train_manual.csv"))
df = df[df.label != 2].copy(); df["label"] = df.label.replace({-1: 0})   # same as e554bad
ds = T.TweetDataset(df.cleaned_text.tolist(), tok, df.label.tolist())
if mode == "full":   # e554bad 80 20
    n = int(len(ds) * T.TRAIN_SPLIT); ds, _ = random_split(ds, [n, len(ds) - n])
loader = DataLoader(ds, batch_size=T.BATCH_SIZE, shuffle=True)
model = M.BERTweetStanceClassifier().to(dev)
if mode == "frozen":
    model.encoder.requires_grad_(False)
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=T.LEARNING_RATE)
else:
    opt = torch.optim.AdamW(model.parameters(), lr=T.LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss(); losses = []
for ep in range(T.EPOCHS):
    model.train()
    for b in loader:
        opt.zero_grad()
        out = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
        loss = loss_fn(out["logits"], b["label"].to(dev)); loss.backward(); opt.step()
        losses.append(loss.item())
# grade
te = pd.read_csv(os.path.join(work, "test_heldout.csv"))
model.eval(); preds = []
with torch.no_grad():
    for i in range(0, len(te), 64):
        enc = M.tokenize_batch(te.cleaned_text.iloc[i:i+64].tolist(), tok)
        out = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
        preds += out["probs"].argmax(-1).cpu().tolist()
te["pred"] = [1 if p == 1 else -1 for p in preds]
pol = te[te.label != 2]
res = dict(commit=os.path.basename(vdir), mode=mode, seed=seed, n_train=len(ds),
           pol_acc=float((pol.pred == pol.label).mean()),
           all3_acc=float((te.pred == te.label).mean()),
           losses=losses, epochs=T.EPOCHS, batch=T.BATCH_SIZE, lr=T.LEARNING_RATE)
json.dump(res, open(os.path.join(work, f"m4_{mode}_{seed}.json"), "w"))
if os.environ.get("SAVE"): torch.save(model.state_dict(), os.environ["SAVE"])
te.to_csv(os.path.join(work, f"preds_m4_{mode}_{seed}.csv"), index=False)
print(mode, seed, "pol_acc=%.3f all_acc=%.3f final_loss=%.3f" % (res["pol_acc"], res["all3_acc"], np.mean(losses[-5:])))
