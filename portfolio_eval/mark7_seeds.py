# one good run
import json, os, sys
import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")
import stage3_training as st
import model as fm
from stance_tags import model_text

full = pd.read_pickle("outputs/cache/stage0_cleaned.pkl")
dev = torch.device("cuda")
TESTS = {"heldout": "data/heldout_test.csv", "shortcut": "data/shortcut_test.csv", "fresh": "data/fresh_probe.csv", "blind": "data/blind_test.csv"}
tok = fm.load_tokenizer()


def grade(net, path):
    d = pd.read_csv(path)
    tx = (d.t if "t" in d else d.cleaned_text).astype(str)
    y = d.y if "y" in d else d.label
    keep = y != 2
    tx, y = tx[keep].tolist(), (y[keep] == 1).astype(int).values
    out = []
    with torch.no_grad():
        for i in range(0, len(tx), 32):
            e = fm.tokenize_batch([model_text(t) or "." for t in tx[i:i + 32]], tok)
            out.append(net(e["input_ids"].to(dev), e["attention_mask"].to(dev))["probs"].float().cpu().numpy())
    p = np.vstack(out)
    return float(((p[:, 1] > p[:, 0]).astype(int) == y).mean())


res = {}
for seed in map(int, sys.argv[1:]):
    torch.manual_seed(seed); np.random.seed(seed)
    st.MODEL_DIR = f"model/mark7_seed{seed}/"
    net = st.train(full, dev).eval()
    res[seed] = {k: grade(net, v) for k, v in TESTS.items()}
    print("SEED", seed, {k: round(v, 3) for k, v in res[seed].items()}, flush=True)
    json.dump(res, open("portfolio_eval/mark7_seeds.json", "w"), indent=1)
    del net; torch.cuda.empty_cache()
