# Political Polarization Classifier

fine-tunes [BERTweet](https://huggingface.co/vinai/bertweet-base) (pretrained on 850M tweets) to classify tweets as pro-biden or pro-trump.

## How It Works

1. `create_manual_labels.py` pulls the first 500 tweets from each csv, strips urls/mentions/hashtags, and dumps them to `data/manual_labels.csv` for hand-labeling (`0` = pro-biden, `1` = pro-trump, plus a `2` for ~90 hand-flagged spam/promo tweets that the classifier doesn't use yet)
2. before scoring, `clean.py` strips urls/mentions and drops any tweet with `<= 5` words, plus duplicate (`user_id`, text) pairs - spam/promo tweets tend to be short and repeated word-for-word, so this thins out a chunk of the users who were only "polarized" because they were bots
3. `train.py` fine-tunes `BERTweetStanceClassifier` (`model.py`) on the ~1000 hand-labeled tweets (spam-labeled rows excluded) for 3 epochs
4. runs the fine-tuned model over every cleaned tweet, producing a `[-1, +1]` stance score per tweet (`P(trump) - P(biden)`)
5. groups by `user_id`, aggregates mean/std stance, and sorts by lowest std ("most polarized")

## Project Layout

```
model.py                  # BERTweetStanceClassifier + tokenizer helpers
train.py                  # fine-tunes the classifier, scores all tweets, aggregates per user
clean.py                  # drops short/duplicate tweets before scoring
create_manual_labels.py   # bootstraps data/manual_labels.csv from the raw csvs
data/manual_labels.csv    # ~1000 hand-labeled tweets used for fine-tuning
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

training is much faster with a gpu - grab the cuda-matched pytorch build from [pytorch.org](https://pytorch.org) separately.

## Usage

drop `trump_tweets.csv` and `biden_tweets.csv` (need `tweet` and `user_id` columns) in the project root, then:

```bash
python train.py
```

prints the most "polarized" users (lowest stance std) and saves per-user stats to `user_scores.csv`.

## Known Issues

- `clean.py`'s short/duplicate filter is a blunt heuristic, not a real spam detector - genuine short tweets get dropped too, and smarter spam still slips through. an actual spam classifier is a later problem.
