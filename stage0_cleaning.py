# stage0_cleaning py split

import re

import pandas as pd
from langdetect import detect, LangDetectException

from config import TRUMP_CSV, BIDEN_CSV, MIN_WORDS, MIN_TWEETS_PER_USER


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_long_enough(text: str) -> bool:
    return len(text.split()) > MIN_WORDS


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        # langdetect chokes on
        return False


def load_and_clean():
    """loads both csvs"""
    trump_df = pd.read_csv(TRUMP_CSV)
    biden_df = pd.read_csv(BIDEN_CSV)
    df = pd.concat([trump_df, biden_df], ignore_index=True)
    df = df.dropna(subset=["tweet", "user_id"])

    df["cleaned_text"] = df["tweet"].apply(clean_text)

    before = len(df)
    df = df[df["cleaned_text"].apply(is_long_enough)]
    print(f"Dropped {before - len(df)} tweets with <= {MIN_WORDS} words")

    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "cleaned_text"])
    print(f"Dropped {before - len(df)} duplicate (user, text) tweets")

    before = len(df)
    df = df[df["cleaned_text"].apply(is_english)]
    print(f"Dropped {before - len(df)} non-English tweets")

    full_df = df

    counts = full_df.groupby("user_id")["cleaned_text"].count()
    valid_users = counts[counts >= MIN_TWEETS_PER_USER].index
    filtered_df = full_df[full_df["user_id"].isin(valid_users)].copy()

    print(f"Full dataset: {len(full_df)} tweets, "
          f"filtered dataset: {len(filtered_df)} tweets "
          f"(users with >= {MIN_TWEETS_PER_USER} tweets)")

    return full_df, filtered_df
