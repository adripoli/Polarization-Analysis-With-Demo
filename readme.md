# Political Polarization Pipeline

basically an ML pipeline that tries to measure how polarized twitter/X users are. fine-tunes [BERTweet](https://huggingface.co/vinai/bertweet-base) on a bunch of `#trump` and `#joebiden` tweets, scores each user on a stance axis, then measures how split the population is with a Gaussian Mixture Model.

## What It Actually Does

you give it two CSVs of tweets (one per hashtag) and it:

1. **cleans** the tweet text (urls, mentions, hashtags, emojis, retweets, dupes - all gone)
2. **tokenizes** with BERTweet's BPE tokenizer (`vinai/bertweet-base`)
3. **encodes** tweets into 768-dim `[CLS]` embeddings using a 12-layer transformer
4. **classifies** each tweet as pro-biden / pro-trump / spam, spits out a stance score in `[-1, +1]`. trained with weak supervision (hashtag = label) plus some hand-labeled tweets mixed in extra
5. **aggregates** per user - mean stance, std, entropy, consistency ratio. spam-heavy accounts get dropped
6. **scores polarization** per user - blend of variance, entropy, and consistency, discounted a bit if they don't have many tweets
7. **analyzes the whole population** - stance histogram, fits a 2-cluster GMM, works out the overlap between clusters (lower = more polarized), checks if polarization correlates with how active someone is

## Project Layout

```
political_polarization/
├── main.py                      # runs the whole thing
├── config.py                    # all the settings live here
├── stage0_cleaning.py           # stage 0: load + clean the tweets
├── model.py                     # stages 1-3: tokenizer + bertweet + classifier head
├── stage3_training.py           # stage 3: fine-tuning + inference
├── stage4_5_user_analysis.py    # stages 4-5: per-user aggregation + polarization score
├── stage6_analysis.py           # stage 6: population plots + GMM fit
├── create_manual_labels.py      # helper for bootstrapping a manual-labels csv
├── requirements.txt
├── trump_tweets.csv             # input (user_id, tweet, created_at, ...)
├── biden_tweets.csv             # input
├── data/manual_labels.csv       # (optional) hand-checked labels
├── model/                       # saved fine-tuned weights
└── outputs/                     # plots, per-user scores, GMM summary
```

## Setup

```bash
pip install -r requirements.txt
```

for GPU training grab the CUDA-matched pytorch build from [pytorch.org](https://pytorch.org) separately. CPU-only works but it's slow - use google colab for the training step if you can.

## Usage

drop `trump_tweets.csv` and `biden_tweets.csv` in the project root, columns matching `COL_USER`, `COL_TEXT`, `COL_TIME` in [config.py](config.py).

```bash
# full pipeline: clean -> train -> score -> analyze
python main.py

# skip training and reuse a model you already fine-tuned
python main.py --skip-training
```

stuff lands in `outputs/`:
- `user_polarization_scores.csv` - per-user metrics + final score
- `stance_histogram.png` - distribution of user mean stance
- `gmm_fit.png` - the fitted 2-cluster mixture with the overlap region
- `polarization_vs_activity.png` - scatter + pearson correlation
- `gmm_summary.csv` - cluster means, stds, weights, overlap coefficient

## Configuration

every hyperparameter (model, batch size, epochs, thresholds, scoring weights, etc) lives in [config.py](config.py). the important ones:

| Setting | Default | Effect |
|---|---|---|
| `MIN_TWEETS_PER_USER` | 3 | users below this get left out of the polarization analysis |
| `SPAM_USER_THRESHOLD` | 0.5 | users with >50% spam tweets get dropped |
| `MANUAL_LABEL_WEIGHT` | 5 | how much to oversample the hand-labeled rows vs weak labels |
| `WEIGHT_STD / WEIGHT_ENTROPY / WEIGHT_CONSISTENCY` | 0.6 / 0.2 / 0.2 | weights in the composite polarization score |
| `GMM_COMPONENTS` | 2 | clusters fit to the population stance distribution |

## How Polarization Gets Scored

a user's polarization score goes up for:
- **low variance** in stance (`1 - std_normalized`) - staying consistent
- **high model confidence** (inverted entropy) - unambiguous political language
- **high consistency ratio** - tweets mostly on one side of the axis

then it's scaled by `log(non_spam_tweet_count)` so someone with 50 real tweets counts for more than someone with 3.

for the whole population, the headline number is the **GMM overlap coefficient** - basically the overlap between the two fitted gaussians. closer to 0 = more bimodal = more polarized.
