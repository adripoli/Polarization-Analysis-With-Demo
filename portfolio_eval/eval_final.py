# grade the saved
import sys, os, json, re
import numpy as np, pandas as pd, torch
P = r"C:\Users\Asus Gaming\Desktop\political_polarization"
sys.path.insert(0, P); os.chdir(P)
from model import BERTweetStanceClassifier, load_tokenizer, tokenize_batch
from config import MIN_CONTENT_WORDS, MIN_CLASSIFIER_CONFIDENCE
from stage4_5_user_analysis import _count_content_words
work = sys.argv[1]
dev = torch.device("cuda")
m = BERTweetStanceClassifier().to(dev)
m.load_state_dict(torch.load(r"model\bertweet_finetuned\best_model.pt", map_location=dev)); m.eval()
tok = load_tokenizer(); out = {}
for name in ["test_heldout", "train_manual"]:
    d = pd.read_csv(os.path.join(work, name + ".csv")); probs = []
    with torch.no_grad():
        for i in range(0, len(d), 64):
            e = tokenize_batch(d.cleaned_text.iloc[i:i+64].tolist(), tok)
            probs.append(m(e["input_ids"].to(dev), e["attention_mask"].to(dev))["probs"].float().cpu().numpy())
    p = np.vstack(probs); am = p.argmax(1)
    d["pred_cls"] = np.array([-1, 1, 2])[am]
    # full final spam
    rule = (am == 2) | (p[:, :2].max(1) < MIN_CLASSIFIER_CONFIDENCE) | (d.cleaned_text.apply(_count_content_words) < MIN_CONTENT_WORDS)
    d["pred_rule"] = np.where(rule, 2, np.where(p[:, 1] > p[:, 0], 1, -1))
    d["pred_pol"] = np.where(p[:, 1] > p[:, 0], 1, -1)
    pol = d[d.label != 2]; sp = d[d.label == 2]
    out[name] = dict(
        n=len(d), n_pol=len(pol), n_spam=len(sp),
        pol_acc=float((pol.pred_pol == pol.label).mean()),
        all3_acc_classifier=float((d.pred_cls == d.label).mean()),
        all3_acc_rules=float((d.pred_rule == d.label).mean()),
        spam_recall_classifier=float((sp.pred_cls == 2).mean()),
        spam_recall_rules=float((sp.pred_rule == 2).mean()),
        political_flagged_as_spam_rules=float((pol.pred_rule == 2).mean()),
        pol_acc_kept=float((pol.pred_rule == pol.label)[pol.pred_rule != 2].mean()),
    )
    if name == "test_heldout": d.to_csv(os.path.join(work, "preds_final.csv"), index=False)
json.dump(out, open(os.path.join(work, "final_eval.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
