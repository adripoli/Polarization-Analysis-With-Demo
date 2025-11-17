# train.py
#
# fine-tunes the BERTweet classifier on the ~1000 hand-labeled tweets
# (data/manual_labels.csv, columns: cleaned_text, label)
#
# first pass: freezing the BERTweet encoder and only training the
# classifier head on top - figured that'd be enough given the encoder is
# already pretrained on tweets

import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import BERTweetStanceClassifier, load_tokenizer, tokenize_batch

MANUAL_LABELS_CSV = "data/manual_labels.csv"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3


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


def train_model():
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
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BERTweetStanceClassifier()

    # freeze the encoder - it's already pretrained on tweets, just train
    # the small head on top of it
    model.encoder.requires_grad_(False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]

            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            loss = loss_fn(output["logits"], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: train_loss={total_loss / len(train_loader):.4f}")

    # train loss is stuck around 0.69 (basically ln(2), i.e. random-chance
    # for 2 classes) no matter how many epochs this runs for. a frozen
    # BERTweet's [CLS] embedding clearly isn't linearly separable enough on
    # its own for the small head to do anything with it - probably need to
    # actually fine-tune the encoder, not just the head

    return model


if __name__ == "__main__":
    train_model()
