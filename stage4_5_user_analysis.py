# stage4_5_user_analysis py stage

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from config import (
    COL_USER,
    WEIGHT_STD, WEIGHT_ENTROPY, WEIGHT_CONSISTENCY,
    SPAM_USER_THRESHOLD, MIN_TWEETS_PER_USER,
)


# stage 4 helpers

def _mean_entropy(probs_biden: np.ndarray,
                  probs_trump: np.ndarray) -> float:
    """average binary shannon"""
    probs = np.column_stack([probs_biden, probs_trump])  # n_tweets 2
    tweet_entropies = scipy_entropy(probs.T, base=2)     # n_tweets
    max_entropy = 1.0   # log2 2 max
    return float(1 - (tweet_entropies.mean() / max_entropy))


def _consistency_ratio(stance_scores: np.ndarray) -> float:
    """of tweets on"""
    n = len(stance_scores)
    if n == 0:
        return 0.0
    n_positive = (stance_scores > 0).sum()
    n_negative = (stance_scores < 0).sum()
    return float(max(n_positive, n_negative) / n)


# stage 4 aggregate

def aggregate_users(scored_df: pd.DataFrame) -> pd.DataFrame:
    """works out per"""
    print("\n=== Stage 4: User-Level Aggregation ===")

    # a tweet counts
    scored_df = scored_df.copy()
    probs = scored_df[["prob_biden", "prob_trump", "prob_spam"]].values
    scored_df["is_spam"] = probs.argmax(axis=1) == 2

    records = []

    for user_id, group in scored_df.groupby(COL_USER):
        tweet_count = len(group)
        spam_ratio  = float(group["is_spam"].mean())

        # only compute stance
        non_spam = group[~group["is_spam"]]
        non_spam_count = len(non_spam)
        if non_spam_count == 0:
            mean_stance = std_stance = m_entropy = consistency = 0.0
        else:
            stance_scores = non_spam["stance"].values
            mean_stance = float(np.mean(stance_scores))
            std_stance  = float(np.std(stance_scores))
            m_entropy   = _mean_entropy(
                non_spam["prob_biden"].values,
                non_spam["prob_trump"].values,
            )
            consistency = _consistency_ratio(stance_scores)

        records.append({
            COL_USER:            user_id,
            "tweet_count":       tweet_count,
            "non_spam_count":    non_spam_count,
            "spam_ratio":        spam_ratio,
            "mean_stance":       mean_stance,
            "std_stance":        std_stance,
            "mean_entropy":      m_entropy,
            "consistency_ratio": consistency,
        })

    user_df = pd.DataFrame(records)

    # drop spam heavy
    before = len(user_df)
    user_df = user_df[user_df["spam_ratio"] <= SPAM_USER_THRESHOLD]
    dropped_spam = before - len(user_df)

    # re check the
    before = len(user_df)
    user_df = user_df[user_df["non_spam_count"] >= MIN_TWEETS_PER_USER]
    user_df = user_df.reset_index(drop=True)
    dropped_thin = before - len(user_df)

    print(f"  Aggregated {before + dropped_spam:,} users")
    print(f"  Dropped {dropped_spam:,} spam-heavy users "
          f"(>{SPAM_USER_THRESHOLD:.0%} spam tweets)")
    print(f"  Dropped {dropped_thin:,} users with <{MIN_TWEETS_PER_USER} non-spam tweets")
    print(f"  Retained {len(user_df):,} users for polarization analysis")
    print(f"  Mean non-spam tweets per user : {user_df['non_spam_count'].mean():.1f}")
    print(f"  Stance distribution  : "
          f"mean={user_df['mean_stance'].mean():.3f}  "
          f"std={user_df['mean_stance'].std():.3f}")

    return user_df


# stage 5 polarization

def compute_polarization(user_df: pd.DataFrame) -> pd.DataFrame:
    """combines the per"""
    print("\n=== Stage 5: Polarization Scoring ===")

    df = user_df.copy()

    # normalize std to
    max_std = df["std_stance"].max()
    if max_std == 0:
        df["std_norm"] = 0.0
    else:
        df["std_norm"] = df["std_stance"] / max_std

    # composite polarization score
    df["polarization_raw"] = (
        WEIGHT_STD         * (1 - df["std_norm"])
        + WEIGHT_ENTROPY   * df["mean_entropy"]
        + WEIGHT_CONSISTENCY * df["consistency_ratio"]
    )

    # tweet count weighting
    log_counts = np.log1p(df["non_spam_count"])
    df["tweet_count_weight"] = log_counts / log_counts.max()

    df["polarization_weighted"] = df["polarization_raw"] * df["tweet_count_weight"]

    # re normalize the
    min_p = df["polarization_weighted"].min()
    max_p = df["polarization_weighted"].max()
    df["polarization_weighted"] = (
        (df["polarization_weighted"] - min_p) / (max_p - min_p + 1e-9)
    )

    # sort by polarization
    df = df.sort_values("polarization_weighted", ascending=False).reset_index(drop=True)

    print(f"  Polarization score stats:")
    print(f"    Mean   : {df['polarization_weighted'].mean():.3f}")
    print(f"    Median : {df['polarization_weighted'].median():.3f}")
    print(f"    Std    : {df['polarization_weighted'].std():.3f}")
    print(f"\n  Top 5 most polarized users:")
    print(df[[COL_USER, "tweet_count", "mean_stance",
              "polarization_weighted"]].head())

    return df
