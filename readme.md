# Political Polarization Scorer

a first pass at guessing how polarized twitter/X users were during the 2020 election, using keyword matching plus distance-weighted adjective sentiment.

## How It Works

give it two csvs of tweets - one pulled via `#trump`-ish hashtags, one via `#biden`-ish hashtags - and it:

1. tokenizes each tweet and finds keyword hits at the word level (`keywords.py`)
2. for each keyword hit, finds the *nearest* positive/negative adjective (`adjectives.py`) within `MAX_ADJECTIVE_DISTANCE` (6 words) and weights it by `1 / (1 + distance)` - so a stray adjective describing a different candidate doesn't get misattributed the way "any adjective anywhere in the tweet" used to
3. sums the per-keyword votes into a single tweet score (`+1` / `-1` / `0`)
4. groups by `user_id` and checks the mean/std of each user's scores to see how "polarized" they are

## Project Layout

```
analyze_tweets.py   # tokenizes, scores tweets, aggregates per user
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

- still can't handle sarcasm or broken grammar
- word-distance is a crude stand-in for "what this adjective is actually describing" - a real parser would help but still wouldn't catch sarcasm
- keyword + adjective matching is pretty much maxed out at this point - time to try something else
