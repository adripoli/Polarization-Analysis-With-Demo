# train.py
#
# fine-tunes the BERTweet classifier on 1000 hand-labeled tweets
# (data/manual_labels.csv, columns: cleaned_text, label), then runs it on
# every tweet in trump_tweets.csv / biden_tweets.csv and works out a
# mean/std stance score per user
#
# way better than the keyword approach - actually gets context, sarcasm
# (mostly), no hand-built keyword list needed
#
# known issue: the most "polarized" users by this (lowest std) are often
# just spam accounts posting the same promo tweet over and over. zero
# variation isn't the same as a real, strong opinion, but the model can't
# tell the difference yet

import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from model import BERTweetStanceClassifier, load_tokenizer, tokenize_batch
from clean import clean_dataframe
from config import (
    TRUMP_CSV, BIDEN_CSV, MANUAL_LABELS_CSV,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_SPLIT,
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
    print("Loading manually labelled tweets...")
    labelled_df = pd.read_csv(MANUAL_LABELS_CSV)

    # a handful of these got labeled 2 (spam) while going through them by
    # hand, but the classifier doesn't have a spam class yet so just skip them
    labelled_df = labelled_df[labelled_df["label"] != 2].copy()
    labelled_df["label"] = labelled_df["label"].replace({-1: 0})
    print(f"  {len(labelled_df)} labelled tweets (spam-labelled rows excluded)")

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
    loss_fn = nn.CrossEntropyLoss()
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


def score_all_tweets(model, device):
    trump_df = pd.read_csv(TRUMP_CSV)
    biden_df = pd.read_csv(BIDEN_CSV)
    df = pd.concat([trump_df, biden_df], ignore_index=True)
    df = df.dropna(subset=["tweet", "user_id"])

    print("Cleaning tweets before scoring...")
    df = clean_dataframe(df)

    tokenizer = load_tokenizer()
    dataset = TweetDataset(df["cleaned_text"].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    stances = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring all tweets"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output = model(input_ids, attention_mask)
            stances.extend(output["stance"].cpu().numpy().tolist())

    df["stance"] = stances
    return df


def aggregate_users(scored_df):
    user_df = scored_df.groupby("user_id")["stance"].agg(["mean", "std", "count"])
    # lowest std = "most polarized" but a lot of these are spam accounts
    # posting identical stuff, not genuinely consistent opinions. need to
    # filter those out somehow
    user_df = user_df.sort_values("std")
    return user_df


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = train_model(device)
    scored_df = score_all_tweets(model, device)
    user_df = aggregate_users(scored_df)

    print("\nMost 'polarized' users (lowest stance std):")
    print(user_df.head(10))

    user_df.to_csv("user_scores.csv")


if __name__ == "__main__":
    main()
