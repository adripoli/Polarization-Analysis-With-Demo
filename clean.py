# clean py cleaning

import re

MIN_WORDS = 5


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_long_enough(text: str) -> bool:
    # off by one
    return len(text.split()) > MIN_WORDS


def clean_dataframe(df):
    df = df.copy()
    df["cleaned_text"] = df["tweet"].apply(clean_text)

    before = len(df)
    df = df[df["cleaned_text"].apply(is_long_enough)]
    print(f"Dropped {before - len(df)} tweets with <= {MIN_WORDS} words")

    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "cleaned_text"])
    print(f"Dropped {before - len(df)} duplicate (user, text) tweets")

    return df
