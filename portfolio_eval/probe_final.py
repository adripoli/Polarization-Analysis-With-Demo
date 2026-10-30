import sys, os, pickle, numpy as np, pandas as pd, torch
P = r"C:\Users\Asus Gaming\Desktop\political_polarization"; sys.path.insert(0, P); os.chdir(P)
from model import BERTweetStanceClassifier, load_tokenizer, tokenize_batch
dev = torch.device("cuda"); m = BERTweetStanceClassifier().to(dev)
m.load_state_dict(torch.load(r"model\bertweet_finetuned\best_model.pt", map_location=dev)); m.eval()
tok = load_tokenizer()
full = pickle.load(open(r"outputs\cache\stage0_cleaned.pkl", "rb")).sample(3000, random_state=0)
probs = []
with torch.no_grad():
    for i in range(0, len(full), 64):
        e = tokenize_batch(full.cleaned_text.iloc[i:i+64].tolist(), tok)
        probs.append(m(e["input_ids"].to(dev), e["attention_mask"].to(dev))["probs"].float().cpu().numpy())
p = np.vstack(probs)
print("mean probs [biden,trump,spam]:", p.mean(0).round(3), " argmax counts:", np.bincount(p.argmax(1), minlength=3))
print("max political prob quantiles:", np.quantile(p[:, :2].max(1), [.05, .5, .95]).round(3))
w = pd.read_csv(r"outputs\user_polarization_scores_2nd_run.csv")
print(w.describe().T[["mean", "50%", "max"]].round(3))
