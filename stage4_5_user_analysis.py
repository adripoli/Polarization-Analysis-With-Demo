# stage4_5_user_analysis py stage

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from config import SPAM_USER_THRESHOLD, MIN_TWEETS_PER_USER

COL_USER = "user_id"


def _mean_entropy(probs_biden, probs_trump):
    """average binary shannon"""
    probs = np.column_stack([probs_biden, probs_trump])
    tweet_entropies = scipy_entropy(probs.T, base=2)
    return float(1 - tweet_entropies.mean())  # max binary entropy


def _consistency_ratio(stance_scores):
    """of tweets on"""
    n = len(stance_scores)
    if n == 0:
        return 0.0
    n_positive = (stance_scores > 0).sum()
    n_negative = (stance_scores < 0).sum()
    return float(max(n_positive, n_negative) / n)


def aggregate_users(scored_df):
    """works out per"""
    scored_df = scored_df.copy()
    probs = scored_df[["prob_biden", "prob_trump", "prob_spam"]].values
    scored_df["is_spam"] = probs.argmax(axis=1) == 2

    records = []
    for user_id, group in scored_df.groupby(COL_USER):
        tweet_count = len(group)
        spam_ratio = float(group["is_spam"].mean())

        non_spam = group[~group["is_spam"]]
        non_spam_count = len(non_spam)
        if non_spam_count == 0:
            mean_stance = std_stance = m_entropy = consistency = 0.0
        else:
            stance_scores = non_spam["stance"].values
            mean_stance = float(np.mean(stance_scores))
            std_stance = float(np.std(stance_scores))
            m_entropy = _mean_entropy(
                non_spam["prob_biden"].values, non_spam["prob_trump"].values
            )
            consistency = _consistency_ratio(stance_scores)

        records.append({
            COL_USER: user_id,
            "tweet_count": tweet_count,
            "non_spam_count": non_spam_count,
            "spam_ratio": spam_ratio,
            "mean_stance": mean_stance,
            "std_stance": std_stance,
            "mean_entropy": m_entropy,
            "consistency_ratio": consistency,
        })

    user_df = pd.DataFrame(records)

    before = len(user_df)
    user_df = user_df[user_df["spam_ratio"] <= SPAM_USER_THRESHOLD]
    print(f"Dropped {before - len(user_df)} spam-heavy users "
          f"(>{SPAM_USER_THRESHOLD:.0%} spam tweets)")

    # re check the
    before = len(user_df)
    user_df = user_df[user_df["non_spam_count"] >= MIN_TWEETS_PER_USER]
    user_df = user_df.reset_index(drop=True)
    print(f"Dropped {before - len(user_df)} users with "
          f"<{MIN_TWEETS_PER_USER} non-spam tweets")

    print(f"Retained {len(user_df)} users for polarization analysis")
    return user_df
