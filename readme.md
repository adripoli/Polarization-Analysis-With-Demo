# Political Polarization Scorer

a first pass at guessing how polarized twitter/X users were during the 2020 election, using nothing fancier than keyword matching.

## How It Works

give it two csvs of tweets - one pulled via `#trump`-ish hashtags, one via `#biden`-ish hashtags - and it:

1. scans each tweet for a hand-picked list of pro-trump / pro-biden hashtags and phrases (`keywords.py`)
2. scores every tweet `+1` (pro-trump), `-1` (pro-biden), or `0` (no keyword match, or both)
3. groups by `user_id` and checks the mean/std of each user's scores to see how "polarized" they are

## Project Layout

```
analyze_tweets.py   # loads the csvs, scores tweets, aggregates per user
keywords.py          # TRUMP_KEYWORDS / BIDEN_KEYWORDS hashtag lists
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

- a huge chunk of tweets score `0` because they don't contain any of the listed keywords
- keyword matches can be wrong even when a keyword IS present - e.g. "those MAGA people are crazy" has "maga" in it but is obviously anti-trump. keyword matching alone can't tell a supportive mention from a mocking one.
