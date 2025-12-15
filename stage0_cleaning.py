# stage0_cleaning py stage

import re
import pandas as pd
from emoji import demojize          # pip install emoji
from config import (
    TRUMP_CSV, BIDEN_CSV,
    COL_USER, COL_TEXT, COL_TIME,
    MIN_TWEETS_PER_USER,
)


# helpers

def _remove_urls(text: str) -> str:
    """strips http https"""
    return re.sub(r"http\S+|www\.\S+", "", text)


def _normalize_mentions(text: str) -> str:
    """swaps username for"""
    return re.sub(r"@\w+", "@USER", text)


def _normalize_hashtags(text: str) -> str:
    """lowercases hashtags so"""
    return re.sub(r"#(\w+)", lambda m: "#" + m.group(1).lower(), text)


def _convert_emojis(text: str) -> str:
    """turns emoji characters"""
    return demojize(text, delimiters=(" ", " "))


def _is_retweet(text: str) -> bool:
    """retweets start with"""
    return text.strip().startswith("RT @")


def clean_text(text: str) -> str:
    """runs the full"""
    text = _remove_urls(text)
    text = _normalize_mentions(text)
    text = _normalize_hashtags(text)
    text = _convert_emojis(text)
    # collapse extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# main function

def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    """loads merges and"""

    # 1 load
    print("Loading CSVs...")
    trump_df = pd.read_csv(TRUMP_CSV, dtype={COL_USER: str})
    biden_df = pd.read_csv(BIDEN_CSV, dtype={COL_USER: str})

    # tag which csv
    trump_df["source"] = "trump"
    biden_df["source"] = "biden"

    df = pd.concat([trump_df, biden_df], ignore_index=True)
    print(f"  Loaded {len(df):,} total rows")

    # 2 make sure
    required = [COL_USER, COL_TEXT]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}. "
            f"Update the COL_* constants in config.py to match your files."
        )

    # 3 drop rows
    before = len(df)
    df = df.dropna(subset=[COL_TEXT, COL_USER])
    print(f"  Dropped {before - len(df):,} rows with missing text/user")

    # 4 drop retweets
    before = len(df)
    df = df[~df[COL_TEXT].astype(str).apply(_is_retweet)]
    print(f"  Dropped {before - len(df):,} retweets")

    # 5 clean the
    print("  Cleaning tweet text (URLs, mentions, hashtags, emojis)...")
    df["cleaned_text"] = df[COL_TEXT].astype(str).apply(clean_text)

    # drop anything empty
    before = len(df)
    df = df[df["cleaned_text"].str.strip().str.len() > 0]
    print(f"  Dropped {before - len(df):,} tweets empty after cleaning")

    # 6 drop duplicate
    before = len(df)
    df = df.drop_duplicates(subset=[COL_USER, "cleaned_text"])
    print(f"  Dropped {before - len(df):,} duplicate (user, text) pairs")

    # 7 build full_df
    full_df = df[[COL_USER, "cleaned_text", "source"]].copy()
    if COL_TIME in df.columns:
        full_df[COL_TIME] = df[COL_TIME]

    print(f"\n  Full cleaned dataset : {len(full_df):,} tweets "
          f"from {full_df[COL_USER].nunique():,} users")

    # 8 build filtered_df
    tweet_counts = full_df.groupby(COL_USER)["cleaned_text"].count()
    valid_users  = tweet_counts[tweet_counts >= MIN_TWEETS_PER_USER].index
    filtered_df  = full_df[full_df[COL_USER].isin(valid_users)].copy()

    print(f"  Filtered dataset     : {len(filtered_df):,} tweets "
          f"from {filtered_df[COL_USER].nunique():,} users "
          f"(>= {MIN_TWEETS_PER_USER} tweets each)\n")

    return full_df, filtered_df
