import pandas as pd

from keywords import TRUMP_KEYWORDS, BIDEN_KEYWORDS

TRUMP_CSV = "trump_tweets.csv"
BIDEN_CSV = "biden_tweets.csv"


def score_tweet(text: str) -> int:
    text_lower = str(text).lower()

    is_trump = any(kw in text_lower for kw in TRUMP_KEYWORDS)
    is_biden = any(kw in text_lower for kw in BIDEN_KEYWORDS)

    if is_trump and not is_biden:
        return 1
    if is_biden and not is_trump:
        return -1
    return 0  # no keyword or


def main():
    trump_df = pd.read_csv(TRUMP_CSV)
    biden_df = pd.read_csv(BIDEN_CSV)
    df = pd.concat([trump_df, biden_df], ignore_index=True)

    print(f"Loaded {len(df)} tweets")

    df["score"] = df["tweet"].apply(score_tweet)

    n_neutral = (df["score"] == 0).sum()
    print(f"Neutral (0) tweets: {n_neutral} / {len(df)} "
          f"({100 * n_neutral / len(df):.1f}%)")


if __name__ == "__main__":
    main()
