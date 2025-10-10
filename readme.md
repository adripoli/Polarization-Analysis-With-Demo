# Political Polarization Scorer

a first pass at guessing how polarized twitter/X users were during the 2020 election, using keyword matching plus a check for obviously positive/negative language.

## How It Works

give it two csvs of tweets - one pulled via `#trump`-ish hashtags, one via `#biden`-ish hashtags - and it:

1. scans each tweet for a (bigger, now) list of pro-trump / pro-biden hashtags and phrases (`keywords.py`)
2. if a keyword hits, checks whether the tweet also contains a clearly positive or negative adjective (`adjectives.py`) - if a negative adjective shows up with no positive one, the keyword's polarity gets flipped. e.g. "those MAGA people are crazy" no longer counts as pro-trump
3. groups by `user_id` and checks the mean/std of each user's scores to see how "polarized" they are

## Project Layout

```
analyze_tweets.py   # loads the csvs, scores tweets, aggregates per user
keywords.py          # TRUMP_KEYWORDS / BIDEN_KEYWORDS hashtag lists
adjectives.py         # POSITIVE_ADJECTIVES / NEGATIVE_ADJECTIVES lists
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

drop `trump_tweets.csv` and `biden_tweets.csv` (need `tweet` and `user_id` columns) in the project root, then:

```bash
python analyze_tweets.py
```

prints the most "varied" users by score std and saves per-user stats to `user_scores.csv`.

## Known Issues

- still a lot of tweets score `0` with no keyword match at all
- words like "crazy" get used as a compliment sometimes too ("crazy good", "crazy talented"), so the flip logic fires when it shouldn't
- a tweet that throws adjectives at multiple candidates at once can't be untangled - the flip applies to the whole tweet, not to a specific keyword
