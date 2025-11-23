# train.py
#
# fine-tunes the BERTweet classifier on the ~1000 hand-labeled tweets
# (data/manual_labels.csv, columns: cleaned_text, label)
#
# freezing the encoder and only training the head didn't work (train loss
# stuck at random-chance) - so now training the whole encoder + head
# together, on gpu if there's one available

import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from model import BERTweetStanceClassifier, load_tokenizer, tokenize_batch

MANUAL_LABELS_CSV = "data/manual_labels.csv"

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_SPLIT = 0.8


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

    # train loss was collapsing to ~0 within an epoch while val loss kept
    # climbing - classic overfitting on ~800 examples. added this split so
    # there's at least a signal for it, and bumped the head's dropout to
    # help
    n_train = int(len(dataset) * TRAIN_SPLIT)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = BERTweetStanceClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    # whole model this time, not just the classifier head
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_model(device)
