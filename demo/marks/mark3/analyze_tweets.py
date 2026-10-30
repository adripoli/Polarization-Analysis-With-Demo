# analyze_tweets.py
#
# third attempt at scoring polarization from tweet text
#
# mark 2 flipped a keyword's score if ANY negative adjective showed up
# anywhere in the tweet. breaks down on stuff like:
#
#   "Incompetent. Just like everyone else involved in the trump clown
#    car. VoteBiden"
#
# "incompetent" is about trump, not the VoteBiden keyword, but mark 2 just
# sees keyword=biden + negative word present -> flips it to pro-trump. wrong.
#
# mark 3 fixes this (kinda) by measuring word-distance: for each keyword,
# grab the CLOSEST adjective instead of "any adjective anywhere", and weight
# it down the further away it is (weight = 1 / (1 + distance))

import re

import pandas as pd

from keywords import TRUMP_KEYWORDS, BIDEN_KEYWORDS
from adjectives import POSITIVE_ADJECTIVES, NEGATIVE_ADJECTIVES

TRUMP_CSV = "trump_tweets.csv"
BIDEN_CSV = "biden_tweets.csv"

MAX_ADJECTIVE_DISTANCE = 6  # words - past this, don't bother attributing it


def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


def _find_indices(tokens: list, vocabulary: list) -> list:
    """indices of tokens that match something in vocabulary"""
    return [i for i, tok in enumerate(tokens) if tok in vocabulary]


def _nearest_adjective_weight(keyword_idx: int, adjective_indices: list) -> float:
    """
    signed weight for the closest adjective to keyword_idx:
      +weight if it's positive, -weight if negative, 0 if nothing's close enough
    """
    if not adjective_indices:
        return 0.0

    nearest_idx = min(adjective_indices, key=lambda i: abs(i - keyword_idx))
    distance = abs(nearest_idx - keyword_idx)

    if distance > MAX_ADJECTIVE_DISTANCE:
        return 0.0

    return 1.0 / (1 + distance)


def score_tweet(text: str) -> int:
    tokens = _tokenize(text)

    trump_idx = _find_indices(tokens, TRUMP_KEYWORDS)
    biden_idx = _find_indices(tokens, BIDEN_KEYWORDS)

    if not trump_idx and not biden_idx:
        return 0

    positive_idx = _find_indices(tokens, POSITIVE_ADJECTIVES)
    negative_idx = _find_indices(tokens, NEGATIVE_ADJECTIVES)

    raw_score = 0.0

    # each keyword hit gets its own vote instead of one flag for the whole tweet
    for idx in trump_idx:
        pos_w = _nearest_adjective_weight(idx, positive_idx)
        neg_w = _nearest_adjective_weight(idx, negative_idx)
        raw_score += 1.0 if pos_w >= neg_w else -1.0

    for idx in biden_idx:
        pos_w = _nearest_adjective_weight(idx, positive_idx)
        neg_w = _nearest_adjective_weight(idx, negative_idx)
        raw_score += -1.0 if pos_w >= neg_w else 1.0

    if raw_score > 0:
        return 1
    if raw_score < 0:
        return -1
    return 0


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

    # still broken on bad grammar, sarcasm, and general outliers. distance is
    # a crude stand-in for "what is this adjective actually about" - a real
    # parser would help but wouldn't catch sarcasm either. time to try
    # something else


if __name__ == "__main__":
    main()
