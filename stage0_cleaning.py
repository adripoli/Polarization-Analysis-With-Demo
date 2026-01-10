# stage0_cleaning py Stage

import os
import pickle
import re
import pandas as pd
from emoji import demojize          # pip install emoji
from langdetect import detect, DetectorFactory, LangDetectException
from tqdm import tqdm
from config import (
    TRUMP_CSV, BIDEN_CSV, STAGE0_CACHE,
    COL_USER, COL_TEXT, COL_TIME,
    MIN_TWEETS_PER_USER, FILTER_LANGUAGE,
)

# langdetect uses a
DetectorFactory.seed = 0


# helpers

def _remove_urls(text: str) -> str:
    """Strip http https"""
    return re.sub(r"http\S+|www\.\S+", "", text)


def _normalize_mentions(text: str) -> str:
    """Replace username with"""
    return re.sub(r"@\w+", "@USER", text)


def _normalize_hashtags(text: str) -> str:
    """Lowercase all hashtags"""
    return re.sub(r"#(\w+)", lambda m: "#" + m.group(1).lower(), text)


def _convert_emojis(text: str) -> str:
    """Convert emoji characters"""
    return demojize(text, delimiters=(" ", " "))


def _is_retweet(text: str) -> bool:
    """Retweets start with"""
    return text.strip().startswith("RT @")


def _detect_lang(text: str) -> str:
    """Return the ISO"""
    try:
        return detect(text)
    except LangDetectException:
        return ""


def clean_text(text: str) -> str:
    """Apply the full"""
    text = _remove_urls(text)
    text = _normalize_mentions(text)
    text = _normalize_hashtags(text)
    text = _convert_emojis(text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# cache helpers

def _cache_is_fresh() -> bool:
    """Cache is valid"""
    if not os.path.exists(STAGE0_CACHE):
        return False
    cache_mtime = os.path.getmtime(STAGE0_CACHE)
    for csv_path in (TRUMP_CSV, BIDEN_CSV):
        if not os.path.exists(csv_path):
            return False
        if os.path.getmtime(csv_path) > cache_mtime:
            return False
    return True


def _build_filtered_df(full_df: pd.DataFrame) -> pd.DataFrame:
    """Derive the analysis"""
    tweet_counts = full_df.groupby(COL_USER)["cleaned_text"].count()
    valid_users  = tweet_counts[tweet_counts >= MIN_TWEETS_PER_USER].index
    return full_df[full_df[COL_USER].isin(valid_users)].copy()


# main public function

def load_and_clean(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load merge and"""

    # Cache hit skip
    if use_cache and _cache_is_fresh():
        print(f"Loading cached Stage 0 output from {STAGE0_CACHE}")
        with open(STAGE0_CACHE, "rb") as f:
            full_df = pickle.load(f)
        print(f"  Cached full dataset  : {len(full_df):,} tweets "
              f"from {full_df[COL_USER].nunique():,} users")

        filtered_df = _build_filtered_df(full_df)
        print(f"  Filtered dataset     : {len(filtered_df):,} tweets "
              f"from {filtered_df[COL_USER].nunique():,} users "
              f"(>= {MIN_TWEETS_PER_USER} tweets each)\n")
        return full_df, filtered_df

    # 1 Load
    print("Loading CSVs...")
    trump_df = pd.read_csv(TRUMP_CSV, dtype={COL_USER: str})
    biden_df = pd.read_csv(BIDEN_CSV, dtype={COL_USER: str})

    # tag the source
    trump_df["source"] = "trump"
    biden_df["source"] = "biden"

    df = pd.concat([trump_df, biden_df], ignore_index=True)
    print(f"  Loaded {len(df):,} total rows")

    # 2 Enforce expected
    required = [COL_USER, COL_TEXT]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}. "
            f"Update the COL_* constants in config.py to match your files."
        )

    # 3 Drop rows
    before = len(df)
    df = df.dropna(subset=[COL_TEXT, COL_USER])
    print(f"  Dropped {before - len(df):,} rows with missing text/user")

    # 4 Remove retweets
    before = len(df)
    df = df[~df[COL_TEXT].astype(str).apply(_is_retweet)]
    print(f"  Dropped {before - len(df):,} retweets")

    # 5 Clean text
    print("  Cleaning tweet text (URLs, mentions, hashtags, emojis)...")
    df["cleaned_text"] = df[COL_TEXT].astype(str).apply(clean_text)

    # drop tweets that
    before = len(df)
    df = df[df["cleaned_text"].str.strip().str.len() > 0]
    print(f"  Dropped {before - len(df):,} tweets empty after cleaning")

    # 5b Filter by
    if FILTER_LANGUAGE:
        print(f"  Detecting tweet language (keeping '{FILTER_LANGUAGE}' only)...")
        tqdm.pandas(desc="  langdetect")
        df["lang"] = df["cleaned_text"].progress_apply(_detect_lang)

        before = len(df)
        df = df[df["lang"] == FILTER_LANGUAGE]
        print(f"  Dropped {before - len(df):,} non-{FILTER_LANGUAGE} tweets")
        df = df.drop(columns=["lang"])

    # 6 Remove duplicate
    before = len(df)
    df = df.drop_duplicates(subset=[COL_USER, "cleaned_text"])
    print(f"  Dropped {before - len(df):,} duplicate (user, text) pairs")

    # 7 Build full_df
    full_df = df[[COL_USER, "cleaned_text", "source"]].copy()
    if COL_TIME in df.columns:
        full_df[COL_TIME] = df[COL_TIME]

    print(f"\n  Full cleaned dataset : {len(full_df):,} tweets "
          f"from {full_df[COL_USER].nunique():,} users")

    # 7b Persist full_df
    os.makedirs(os.path.dirname(STAGE0_CACHE), exist_ok=True)
    with open(STAGE0_CACHE, "wb") as f:
        pickle.dump(full_df, f)
    print(f"  Cached Stage 0 output to {STAGE0_CACHE}")

    # 8 Build filtered_df
    filtered_df = _build_filtered_df(full_df)

    print(f"  Filtered dataset     : {len(filtered_df):,} tweets "
          f"from {filtered_df[COL_USER].nunique():,} users "
          f"(>= {MIN_TWEETS_PER_USER} tweets each)\n")

    return full_df, filtered_df
