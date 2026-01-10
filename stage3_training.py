# ── stage3_training.py ─────────────────────────────────────────────────────
# Stage 3: Fine-tuning & Inference
#
# Two responsibilities:
#   (a) TRAINING  — fine-tune the BERTweet classifier on labelled tweets
#   (b) INFERENCE — run the trained model on all 750k tweets to produce
#                   a stance score [-1, +1] for every tweet
#
# Labelling strategy (weak supervision):
#   Since manually labelling 750k tweets is infeasible, we use the hashtag
#   as a weak label:
#     • Tweets containing ONLY #trump  → weakly labelled Pro-Trump  (class 1)
#     • Tweets containing ONLY #joebiden → weakly labelled Pro-Biden (class 0)
#     • A small hand-labelled set (~hundreds) is added for ground truth.
#
# Data-integrity rules (see README of this file):
#   1. The train/val split is STRATIFIED on label (so rare classes like Spam
#      are represented in both halves).
#   2. Manual labels are oversampled (×MANUAL_LABEL_WEIGHT) AFTER the split,
#      and ONLY on the training set — preventing leakage that would otherwise
#      put duplicates of the same manual tweet in both train and val.
# ───────────────────────────────────────────────────────────────────────────

import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, HEAD_LEARNING_RATE, TRAIN_SPLIT,
    NUM_LABELS, MODEL_DIR, MAX_LENGTH,
    MANUAL_LABELS_CSV, MANUAL_LABEL_WEIGHT,
    LABEL_SMOOTHING, EARLY_STOPPING_PATIENCE,
    USE_AMP, NUM_WORKERS, INFERENCE_BATCH_SIZE,
)
from model import BERTweetStanceClassifier, load_tokenizer


# ── Dataset ────────────────────────────────────────────────────────────────

class TweetDataset(Dataset):
    """
    PyTorch Dataset over pre-tokenized tweets.

    Pre-tokenizing once upfront (rather than re-running the slow BERTweet
    tokenizer inside every __getitem__ call) is dramatically faster on a
    multi-epoch run — the slow tokenizer was the dominant cost previously.

    Parameters
    ----------
    input_ids      : LongTensor [N, MAX_LENGTH]
    attention_mask : LongTensor [N, MAX_LENGTH]
    labels         : LongTensor [N] or None (None = inference, no labels)
    """

    def __init__(self, input_ids, attention_mask, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        item = {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def pre_tokenize(texts: list[str], tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize an entire list of tweets into padded tensors in one pass."""
    encoded = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


# ── Weak labelling ─────────────────────────────────────────────────────────

def assign_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a weak training label to each tweet based on its source hashtag.

    Label map:
        0 → Pro-Biden  (tweet came from the #joebiden CSV)
        1 → Pro-Trump  (tweet came from the #trump CSV)
    """
    label_map = {"biden": 0, "trump": 1}

    df = df.copy()
    df["label"] = df["source"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["is_manual"] = False

    print(f"Weak labels assigned:")
    print(f"  Pro-Biden  : {(df['label'] == 0).sum():,}")
    print(f"  Pro-Trump  : {(df['label'] == 1).sum():,}")
    return df


# ── Manual labels ──────────────────────────────────────────────────────────

def load_manual_labels(path: str) -> pd.DataFrame:
    """
    Load a CSV of manually verified tweets. Returns a single copy per row —
    oversampling happens later, AFTER the train/val split, in train().

    Expected columns:
        cleaned_text : the cleaned tweet string (same format as Stage 0 output)
        label        : integer  0=Pro-Biden  1=Pro-Trump  2=Spam
                       (-1 is also accepted and remapped to 0 for Biden)
    """
    df = pd.read_csv(path)
    required = {"cleaned_text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manual_labels.csv is missing columns: {missing}")

    df["label"] = df["label"].astype(int)
    df.loc[df["label"] == -1, "label"] = 0

    valid_labels = {0, 1, 2}
    invalid = set(df["label"].unique()) - valid_labels
    if invalid:
        raise ValueError(
            f"manual_labels.csv contains invalid labels: {invalid}. "
            f"Expected one of 0 (Biden), 1 (Trump), 2 (Spam), or -1 (Biden)."
        )

    df["is_manual"] = True

    label_counts = df["label"].value_counts().sort_index()
    print(f"Manual labels loaded (single copy — oversampling applied post-split):")
    for cls, name in zip([0, 1, 2], ["Pro-Biden", "Pro-Trump", "Spam"]):
        print(f"  {name}: {label_counts.get(cls, 0):,}")
    return df


# ── Training ───────────────────────────────────────────────────────────────

def _macro_f1(y_true, y_pred) -> float:
    return f1_score(
        y_true, y_pred,
        labels=list(range(NUM_LABELS)),
        average="macro",
        zero_division=0,
    )


def train(full_df: pd.DataFrame, device: torch.device) -> BERTweetStanceClassifier:
    """
    Fine-tune BERTweet on weakly labelled tweets + manual labels.

    Parameters
    ----------
    full_df : the FULL cleaned dataframe from Stage 0 (not the filtered one)
              — we want as many training examples as possible
    device  : torch.device (cuda or cpu)

    Returns
    -------
    Trained BERTweetStanceClassifier (with best-checkpoint weights reloaded)
    """
    print("\n=== Stage 3: Training ===")

    tokenizer = load_tokenizer()

    # ── 1. assemble labelled data (single copy of manual rows, no leakage) ─
    weak_df = assign_weak_labels(full_df)[["cleaned_text", "label", "is_manual"]]

    if MANUAL_LABELS_CSV and os.path.exists(MANUAL_LABELS_CSV):
        manual_df = load_manual_labels(MANUAL_LABELS_CSV)[
            ["cleaned_text", "label", "is_manual"]
        ]
        combined = pd.concat([weak_df, manual_df], ignore_index=True)
    else:
        combined = weak_df
        if MANUAL_LABELS_CSV:
            print(f"  (manual labels file {MANUAL_LABELS_CSV} not found — "
                  f"training on weak labels only)")

    combined = combined.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
    print(f"Combined dataset: {len(combined):,} rows "
          f"({combined['is_manual'].sum():,} manual)")

    # ── 2. STRATIFIED train/val split (fixes imbalanced minority class) ────
    # Every manual label appears in exactly ONE side of the split at this
    # point — oversampling happens *after* the split, training-only.
    train_df, val_df = train_test_split(
        combined,
        train_size=TRAIN_SPLIT,
        stratify=combined["label"],
        random_state=42,
    )
    print(f"Split: train={len(train_df):,}  val={len(val_df):,}")
    _print_class_distribution("  val class distribution", val_df["label"])

    # ── 3. Oversample manual rows in the TRAINING SET ONLY ─────────────────
    # This is the fix for validation leakage. Manual rows in train_df get
    # duplicated MANUAL_LABEL_WEIGHT times; val_df is untouched.
    manual_train = train_df[train_df["is_manual"]]
    if len(manual_train) > 0 and MANUAL_LABEL_WEIGHT > 1:
        extra = pd.concat(
            [manual_train] * (MANUAL_LABEL_WEIGHT - 1),
            ignore_index=True,
        )
        train_df = pd.concat([train_df, extra], ignore_index=True)
        print(f"Oversampled manual rows ×{MANUAL_LABEL_WEIGHT} in train "
              f"→ {len(train_df):,} rows")
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── 4. Pre-tokenize both splits once (replaces per-__getitem__ tokenize)
    print("Pre-tokenizing splits...")
    train_ids, train_mask = pre_tokenize(train_df["cleaned_text"].tolist(), tokenizer)
    val_ids,   val_mask   = pre_tokenize(val_df["cleaned_text"].tolist(),   tokenizer)
    train_labels = torch.tensor(train_df["label"].values, dtype=torch.long)
    val_labels   = torch.tensor(val_df["label"].values,   dtype=torch.long)

    train_set = TweetDataset(train_ids, train_mask, train_labels)
    val_set   = TweetDataset(val_ids,   val_mask,   val_labels)

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    # ── 5. Model + optimiser (discriminative LRs) ──────────────────────────
    model = BERTweetStanceClassifier().to(device)

    # Inverse-frequency class weights on the TRAINING set to offset imbalance
    # between weakly labelled Biden/Trump and hand-labelled Spam.
    #
    # The raw "balanced" formula gives Spam a weight ~10,000x the political
    # classes (because Spam only appears in ~hundreds of manual labels vs
    # millions of weak Biden/Trump). That ratio causes the model to collapse
    # and predict Spam for every tweet — the spam filter then drops every
    # user from the polarization analysis. We cap the max weight to 5x the
    # min so Spam is still upweighted, but not catastrophically.
    label_arr = train_df["label"].values
    class_counts = np.bincount(label_arr, minlength=NUM_LABELS).astype(float)
    class_counts[class_counts == 0] = 1.0
    class_weights = len(label_arr) / (NUM_LABELS * class_counts)
    cap = class_weights.min() * 5.0
    class_weights = np.clip(class_weights, None, cap)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    print(f"Class weights (Biden/Trump/Spam, capped at 5×min): "
          f"{class_weights.cpu().numpy().round(3).tolist()}")

    # Label smoothing softens one-hot targets toward a uniform distribution,
    # which is a standard defence against the ~20–30 % label noise that's
    # baked into hashtag-based weak supervision (anti-Trump tweets using
    # #trump, news aggregators tagging both sides, etc.).
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    # Different LR for pretrained encoder (small updates) vs from-scratch
    # head (larger updates). This typically gains 1–3 F1 on top of a single-LR
    # AdamW on small fine-tuning datasets.
    encoder_params = list(model.encoder.parameters())
    head_params    = [p for n, p in model.named_parameters()
                      if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": LEARNING_RATE},
            {"params": head_params,    "lr": HEAD_LEARNING_RATE},
        ],
        weight_decay=0.01,
    )

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = USE_AMP and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("Mixed precision (fp16 autocast) enabled")

    # ── 6. Training loop w/ early stopping on val macro-F1 ─────────────────
    best_val_f1  = -1.0
    epochs_since_improve = 0
    best_path = os.path.join(MODEL_DIR, "best_model.pt")
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        # ── training ───────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels_batch   = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                output = model(input_ids, attention_mask)
                loss   = loss_fn(output["logits"], labels_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ── validation ─────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        all_preds, all_true = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]  "):
                input_ids      = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels_batch   = batch["label"].to(device, non_blocking=True)

                with autocast(enabled=use_amp):
                    output = model(input_ids, attention_mask)
                    loss   = loss_fn(output["logits"], labels_batch)

                total_val_loss += loss.item()
                preds = output["logits"].argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(labels_batch.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = _macro_f1(all_true, all_preds)

        print(f"\nEpoch {epoch+1}: train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  val_macroF1={val_f1:.4f}")
        print(classification_report(
            all_true, all_preds,
            labels=list(range(NUM_LABELS)),
            target_names=["Pro-Biden", "Pro-Trump", "Spam"],
            zero_division=0,
        ))

        # Select on macro-F1, not weighted val loss: the loss is shaped by
        # class_weights and doesn't reflect balanced performance on the
        # under-represented Spam class.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_since_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best model saved (val_macroF1={best_val_f1:.4f})")
        else:
            epochs_since_improve += 1
            print(f"  no improvement ({epochs_since_improve}/{EARLY_STOPPING_PATIENCE})")
            if epochs_since_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"Reloaded best checkpoint (val_macroF1={best_val_f1:.4f})")
    return model


def _print_class_distribution(prefix: str, labels: pd.Series) -> None:
    counts = labels.value_counts().sort_index()
    names = {0: "Biden", 1: "Trump", 2: "Spam"}
    pieces = [f"{names.get(c, c)}={counts.get(c, 0):,}" for c in range(NUM_LABELS)]
    print(f"{prefix}: " + "  ".join(pieces))


# ── Inference ──────────────────────────────────────────────────────────────

def run_inference(
    df: pd.DataFrame,
    model: BERTweetStanceClassifier,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run the trained model on every tweet in df and attach a stance score.

    Returns
    -------
    df with new columns:
        'prob_biden'  : P(Pro-Biden)
        'prob_trump'  : P(Pro-Trump)
        'prob_spam'   : P(Spam)
        'stance'      : P(Trump) - P(Biden)  renormalized over political mass
    """
    print("\n=== Stage 3: Inference ===")

    tokenizer = load_tokenizer()
    input_ids, attention_mask = pre_tokenize(df["cleaned_text"].tolist(), tokenizer)
    dataset = TweetDataset(input_ids, attention_mask, labels=None)
    loader  = DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    use_amp = USE_AMP and device.type == "cuda"
    all_probs, all_stance = [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring tweets"):
            input_ids_b      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask_b = batch["attention_mask"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                output = model(input_ids_b, attention_mask_b)

            all_probs.append(output["probs"].float().cpu().numpy())
            all_stance.append(output["stance"].float().cpu().numpy())

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
