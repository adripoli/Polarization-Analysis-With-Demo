# load the 1
import pandas as pd, sys
P = r"C:\Users\Asus Gaming\Desktop\political_polarization"
t = pd.read_csv(P + r"\trump_tweets.csv", low_memory=False); t["source"]="trump"
b = pd.read_csv(P + r"\biden_tweets.csv", low_memory=False); b["source"]="biden"
df = pd.concat([t,b], ignore_index=True)
df.columns = [c.strip() for c in df.columns]
df[["tweet","user_id","source"]].to_pickle(sys.argv[1])
print(len(df))
