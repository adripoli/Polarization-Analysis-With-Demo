# apply mark 5
import sys, os, time, pandas as pd
from multiprocessing import Pool
V = sys.argv[1]; work = sys.argv[2]
sys.path.insert(0, V)
import clean as C
from langdetect import DetectorFactory

def init():
    DetectorFactory.seed = 0

def eng(t):
    return C.is_english(t)

if __name__ == "__main__":
    t0 = time.time()
    raw = pd.read_pickle(os.path.join(work, "raw.pkl")).dropna(subset=["tweet", "user_id"])
    raw["user_id"] = raw["user_id"].astype(str)
    raw["cleaned_text"] = raw["tweet"].apply(C.clean_text)
    raw["long_enough"] = raw["cleaned_text"].apply(C.is_long_enough)
    raw["is_dup"] = raw.duplicated(subset=["user_id", "cleaned_text"])
    cand = raw.loc[raw.long_enough & ~raw.is_dup, "cleaned_text"].unique()
    print("rows", len(raw), "langdetect on unique texts:", len(cand), flush=True)
    with Pool(os.cpu_count(), initializer=init) as p:
        flags = p.map(eng, cand, chunksize=2000)
    lut = dict(zip(cand, flags))
    raw["is_english"] = raw["cleaned_text"].map(lut).fillna(False).astype(bool)
    raw.drop(columns=["tweet"]).to_pickle(os.path.join(work, "cleaned_flags.pkl"))
    m5 = raw[raw.long_enough & ~raw.is_dup]
    print("raw", len(raw), "| <=5 words dropped", (~raw.long_enough).sum(),
          "| dups dropped", (raw.long_enough & raw.is_dup).sum(),
          "| after mark5", len(m5), "| non-english dropped", (~m5.is_english).sum(),
          "| after mark6", m5.is_english.sum(), "| %.0fs" % (time.time() - t0))
