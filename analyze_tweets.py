# analyze_tweets py first

import re

import pandas as pd

from keywords import TRUMP_KEYWORDS, BIDEN_KEYWORDS
from adjectives import POSITIVE_ADJECTIVES, NEGATIVE_ADJECTIVES

TRUMP_CSV = "trump_tweets.csv"
BIDEN_CSV = "biden_tweets.csv"

MAX_ADJECTIVE_DISTANCE = 6  # words past this


def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


def _find_indices(tokens: list, vocabulary: list) -> list:
    """indices of tokens"""
    return [i for i, tok in enumerate(tokens) if tok in vocabulary]


def _nearest_adjective_weight(keyword_idx: int, adjective_indices: list) -> float:
    """signed weight for"""
    if not adjective_indices:
        return 0.0

    nearest_idx = min(adjective_indices, key=lambda i: abs(i - keyword_idx))
    distance = abs(nearest_idx - keyword_idx)

    if distance > MAX_ADJECTIVE_DISTANCE:
        return 0.0

    return 1.0 / (1 + distance)


def score_tweet(text: str) -> int:
    text_lower = str(text).lower()

    is_trump = any(kw in text_lower for kw in TRUMP_KEYWORDS)
    is_biden = any(kw in text_lower for kw in BIDEN_KEYWORDS)

    if is_trump and is_biden:
        return 0  # mentions both no
    if not is_trump and not is_biden:
        return 0  # no keyword at

    base_score = 1 if is_trump else -1

    # does the tweet
    has_negative = any(adj in text_lower for adj in NEGATIVE_ADJECTIVES)
    has_positive = any(adj in text_lower for adj in POSITIVE_ADJECTIVES)

    if has_negative and not has_positive:
        return -base_score
    return base_score


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
