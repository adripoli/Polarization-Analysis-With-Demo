# analyze_tweets.py
#
# first attempt at scoring political polarization from tweet text
#
# idea: pro-trump tweets usually have certain hashtags (#VoteTrump,
# #Trump2020...) and pro-biden tweets have different ones (#VoteBiden,
# #Biden2020...). so just scan every tweet for these and score it:
#
#   +1 -> has a pro-trump keyword
#   -1 -> has a pro-biden keyword
#    0 -> neither (or both, can't tell)
#
# then group by user and check the mean/spread of their scores to see how
# "polarized" they are

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
    return 0  # no keyword, or both keywords (can't tell which)


def main():
    trump_df = pd.read_csv(TRUMP_CSV)
    biden_df = pd.read_csv(BIDEN_CSV)
    df = pd.concat([trump_df, biden_df], ignore_index=True)

    print(f"Loaded {len(df)} tweets")

    df["score"] = df["tweet"].apply(score_tweet)

    n_neutral = (df["score"] == 0).sum()
    print(f"Neutral (0) tweets: {n_neutral} / {len(df)} "
          f"({100 * n_neutral / len(df):.1f}%)")

    user_scores = df.groupby("user_id")["score"].agg(["mean", "std", "count"])
    user_scores = user_scores.sort_values("std", ascending=False)

    print("\nMost 'varied' users (highest std of score):")
    print(user_scores.head(10))

    user_scores.to_csv("user_scores.csv")
    print("\nSaved per-user scores to user_scores.csv")


if __name__ == "__main__":
    main()
