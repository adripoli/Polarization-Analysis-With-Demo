# stage3_training.py
# stage 3: fine-tuning + inference
#
# two jobs here:
#   (a) TRAINING  - fine-tune the BERTweet classifier on labeled tweets
#   (b) INFERENCE - run the trained model on all 750k tweets to get a
#                   stance score [-1, +1] for every one of them
#
# labeling strategy (weak supervision): manually labeling 750k tweets isn't
# happening, so the hashtag doubles as a weak label:
#   - tweets with ONLY #trump    -> weak label pro-trump (class 1)
#   - tweets with ONLY #joebiden -> weak label pro-biden (class 0)
#   - plus a small hand-labeled set (~200 tweets) mixed in for real signal

import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from tqdm import tqdm

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_SPLIT,
    NUM_LABELS, MODEL_DIR,
    MANUAL_LABELS_CSV, MANUAL_LABEL_WEIGHT,
)
from model import BERTweetStanceClassifier, load_tokenizer, tokenize_batch


# dataset

class TweetDataset(Dataset):
    """wraps a dataframe of cleaned tweets for pytorch.

    df: dataframe with a 'cleaned_text' column
    tokenizer: BERTweet tokenizer
    labels: list/array of int class labels (0=biden, 1=trump, 2=spam), or
            None if this is just for inference"""

    def __init__(self, df: pd.DataFrame, tokenizer, labels=None):
        self.texts     = df["cleaned_text"].tolist()
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = tokenize_batch([self.texts[idx]], self.tokenizer)
        item = {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# weak labeling

def assign_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """gives each tweet a weak training label based on which csv it came from.

    0 = pro-biden (came from the #joebiden csv)
    1 = pro-trump (came from the #trump csv)"""
    # 'source' column got added back in stage0_cleaning.py
    label_map = {"biden": 0, "trump": 1}

    df = df.copy()
    df["label"] = df["source"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print(f"Weak labels assigned:")
    print(f"  Pro-Biden  : {(df['label'] == 0).sum():,}")
    print(f"  Pro-Trump  : {(df['label'] == 1).sum():,}")
    return df


# manual labels

def load_manual_labels(path: str) -> pd.DataFrame:
    """loads the csv of tweets i manually checked.

    expects columns: cleaned_text (same format as stage 0 output) and
    label (0=pro-biden, 1=pro-trump, 2=spam)

    rows get repeated MANUAL_LABEL_WEIGHT times so they outweigh the noisy
    weak labels during training"""
    df = pd.read_csv(path)
    required = {"cleaned_text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manual_labels.csv is missing columns: {missing}")

    df["label"] = df["label"].astype(int)

    # remap -1 -> 0 for biden (csv uses -1/+1/2, model wants 0/1/2)
    df.loc[df["label"] == -1, "label"] = 0

    valid_labels = {0, 1, 2}
    invalid = set(df["label"].unique()) - valid_labels
    if invalid:
        raise ValueError(
            f"manual_labels.csv contains invalid labels: {invalid}. "
            f"Expected one of 0 (Biden), 1 (Trump), 2 (Spam), or -1 (Biden)."
        )

    df = pd.concat([df] * MANUAL_LABEL_WEIGHT, ignore_index=True)

    label_counts = df["label"].value_counts().sort_index()
    print(f"Manual labels loaded (×{MANUAL_LABEL_WEIGHT} oversampling):")
    for cls, name in zip([0, 1, 2], ["Pro-Biden", "Pro-Trump", "Spam"]):
        print(f"  {name}: {label_counts.get(cls, 0):,}")
    return df


# training

def train(full_df: pd.DataFrame, device: torch.device) -> BERTweetStanceClassifier:
    """fine-tunes BERTweet on the weakly labeled tweets.

    full_df should be the FULL cleaned dataframe from stage 0 (not the
    filtered one) - we want as many training examples as we can get.
    returns the trained classifier"""
    print("\n=== Stage 3: Training ===")

    tokenizer = load_tokenizer()
    labelled_df = assign_weak_labels(full_df)

    if MANUAL_LABELS_CSV:
        manual_df = load_manual_labels(MANUAL_LABELS_CSV)
        labelled_df = pd.concat(
            [labelled_df, manual_df[["cleaned_text", "label"]]],
            ignore_index=True,
        ).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Combined dataset: {len(labelled_df):,} rows")

    labels = labelled_df["label"].tolist()
    dataset = TweetDataset(labelled_df, tokenizer, labels)

    # train / val split
    n_train = int(len(dataset) * TRAIN_SPLIT)
    n_val   = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)

    # model, loss, optimizer
    model = BERTweetStanceClassifier().to(device)

    # inverse-frequency class weights to offset the huge imbalance between
    # weak-labeled biden/trump (millions) and hand-labeled spam (hundreds).
    # w_c = N / (K * n_c), the standard sklearn "balanced" formula
    label_arr   = np.array(labels)
    class_counts = np.bincount(label_arr, minlength=NUM_LABELS).astype(float)
    class_counts[class_counts == 0] = 1.0   # avoid divide by zero
    class_weights = len(label_arr) / (NUM_LABELS * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    print(f"Class weights (Biden/Trump/Spam): "
          f"{class_weights.cpu().numpy().round(3).tolist()}")

    # cross entropy wants raw logits + integer labels
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # linear warmup: ramp LR up for the first 10% of steps then decay it.
    # helps stabilize fine-tuning a transformer
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # training loop
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["label"].to(device)

            optimizer.zero_grad()

            output = model(input_ids, attention_mask)
            loss   = loss_fn(output["logits"], labels_batch)
            loss.backward()

            # clip gradients so they don't explode
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # validation loop
        model.eval()
        total_val_loss = 0
        all_preds, all_true = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]  "):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch   = batch["label"].to(device)

                output = model(input_ids, attention_mask)
                loss   = loss_fn(output["logits"], labels_batch)
                total_val_loss += loss.item()

                preds = output["logits"].argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(labels_batch.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}: train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}")
        print(classification_report(
            all_true, all_preds,
            labels=list(range(NUM_LABELS)),
            target_names=["Pro-Biden", "Pro-Trump", "Spam"],
            zero_division=0,
        ))

        # save the best checkpoint so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")

    # load the best weights back before returning
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=device)
    )
    return model


# inference

def run_inference(
    df: pd.DataFrame,
    model: BERTweetStanceClassifier,
    device: torch.device,
) -> pd.DataFrame:
    """runs the trained model over every tweet in df and attaches a stance score.

    gets called TWICE from main.py:
      1. on full_df     -> scores every tweet for training diagnostics
      2. on filtered_df -> these scores feed the polarization analysis

    adds columns: prob_biden, prob_trump, prob_spam, and stance
    (= P(trump) - P(biden), in [-1, +1])"""
    print("\n=== Stage 3: Inference ===")

    tokenizer = load_tokenizer()
    dataset   = TweetDataset(df, tokenizer, labels=None)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs  = []
    all_stance = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring tweets"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output = model(input_ids, attention_mask)

            all_probs.append(output["probs"].cpu().numpy())
            all_stance.append(output["stance"].cpu().numpy())

    probs  = np.vstack(all_probs)        # [N, 3]
    stance = np.concatenate(all_stance)  # [N]

    result_df = df.copy()
    result_df["prob_biden"] = probs[:, 0]
    result_df["prob_trump"] = probs[:, 1]
    result_df["prob_spam"]  = probs[:, 2]
    result_df["stance"]     = stance

    print(f"  Scored {len(result_df):,} tweets")
    print(f"  Stance range: [{stance.min():.3f}, {stance.max():.3f}]  "
          f"mean={stance.mean():.3f}")

    return result_df
