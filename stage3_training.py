# stage3_training.py
#
# fine-tunes the BERTweet classifier (now 3-way: biden/trump/spam) on the
# hand-labeled tweets in data/manual_labels.csv, then scores the
# MIN_TWEETS_PER_USER-filtered tweets from stage0_cleaning so stage4_5 can
# aggregate them per user
#
# way better than the keyword approach - actually gets context, sarcasm
# (mostly), no hand-built keyword list needed
#
# spam-heavy users used to sneak through as "most polarized" (zero std
# looks like a strong consistent opinion, but it's often just a promo bot).
# now that the classifier can actually flag spam tweets, stage4_5 drops
# those users properly instead of relying on the old length/dedupe proxy

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from model import BERTweetStanceClassifier, load_tokenizer, tokenize_batch
from stage0_cleaning import load_and_clean
from stage4_5_user_analysis import aggregate_users
from config import (
    MANUAL_LABELS_CSV,
    MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_SPLIT, NUM_LABELS,
)


class TweetDataset(Dataset):
    def __init__(self, texts, tokenizer, labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = tokenize_batch([self.texts[idx]], self.tokenizer)
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_model(device):
    print(f"Loading manually labelled tweets (tokenizing up to {MAX_LENGTH} tokens)...")
    labelled_df = pd.read_csv(MANUAL_LABELS_CSV)

    # now that the classifier actually has a spam class (NUM_LABELS=3), stop
    # throwing away the label==2 rows - they're the only real spam signal we
    # have
    labelled_df["label"] = labelled_df["label"].replace({-1: 0})
    print(f"  {len(labelled_df)} labelled tweets "
          f"({(labelled_df['label'] == 2).sum()} spam)")

    tokenizer = load_tokenizer()
    dataset = TweetDataset(
        labelled_df["cleaned_text"].tolist(),
        tokenizer,
        labelled_df["label"].tolist(),
    )

    n_train = int(len(dataset) * TRAIN_SPLIT)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = BERTweetStanceClassifier().to(device)

    # spam is maybe 5-10% of the labelled rows, so an unweighted loss barely
    # penalizes missing it. inverse-frequency weighting (sklearn's
    # "balanced" formula) evens that out
    label_counts = labelled_df["label"].value_counts().reindex(
        range(NUM_LABELS), fill_value=0
    ).values.astype(float)
    label_counts[label_counts == 0] = 1.0
    class_weights = len(labelled_df) / (NUM_LABELS * label_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    print(f"  Class weights (biden/trump/spam): {class_weights.cpu().numpy().round(3)}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            loss = loss_fn(output["logits"], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: train_loss={total_loss / len(train_loader):.4f}")

    return model


def score_all_tweets(model, device, df):
    """runs the trained model over every tweet in df, attaching prob_biden/
    prob_trump/prob_spam + stance columns. df should be the
    MIN_TWEETS_PER_USER-filtered dataframe from stage0_cleaning, since
    that's the set the per-user analysis in stage4_5 runs on"""
    tokenizer = load_tokenizer()
    dataset = TweetDataset(df["cleaned_text"].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    probs_all, stances = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring tweets"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output = model(input_ids, attention_mask)
            probs_all.append(output["probs"].cpu().numpy())
            stances.extend(output["stance"].cpu().numpy().tolist())

    probs = np.vstack(probs_all)
    df = df.copy()
    df["prob_biden"] = probs[:, 0]
    df["prob_trump"] = probs[:, 1]
    df["prob_spam"] = probs[:, 2]
    df["stance"] = stances
    return df


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = train_model(device)

    print("Cleaning tweets before scoring...")
    full_df, filtered_df = load_and_clean()
    scored_df = score_all_tweets(model, device, filtered_df)

    user_df = aggregate_users(scored_df)
    user_df = user_df.sort_values("std_stance")

    print("\nMost 'polarized' users (lowest stance std):")
    print(user_df.head(10))

    user_df.to_csv("user_scores.csv", index=False)


if __name__ == "__main__":
    main()
