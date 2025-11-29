# Political Polarization Classifier

fine-tunes [BERTweet](https://huggingface.co/vinai/bertweet-base) (pretrained on 850M tweets) to classify tweets as pro-biden or pro-trump, replacing the keyword/adjective heuristics that hit their ceiling on sarcasm and bad grammar.

## How It Works

1. `create_manual_labels.py` pulls the first 500 tweets from each csv, strips urls/mentions/hashtags, and dumps them to `data/manual_labels.csv` for hand-labeling (`0` = pro-biden, `1` = pro-trump). while labeling by hand, a spam label (`2`) also got used for ~90 obviously promotional tweets, though the classifier doesn't use that class yet
2. `train.py` fine-tunes `BERTweetStanceClassifier` (`model.py`) on those ~1000 hand-labeled tweets (spam-labeled rows excluded) for 3 epochs
3. runs the fine-tuned model over every tweet in `trump_tweets.csv` / `biden_tweets.csv`, producing a `[-1, +1]` stance score per tweet (`P(trump) - P(biden)`)
4. groups by `user_id`, aggregates mean/std stance, and sorts by lowest std ("most polarized")

## Project Layout

```
model.py                  # BERTweetStanceClassifier + tokenizer helpers
train.py                  # fine-tunes the classifier, scores all tweets, aggregates per user
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

- the most "polarized" users by this measure are often just spam accounts posting the same promotional tweet over and over - zero variance isn't the same as a genuinely consistent opinion, and the classifier has no way to tell the difference yet
