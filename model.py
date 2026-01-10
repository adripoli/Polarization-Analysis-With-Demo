# ── model.py ───────────────────────────────────────────────────────────────
# Stages 1, 2 & 3 (model architecture):
#   Stage 1 — BERTweet tokenizer (converts cleaned text → token ID tensors)
#   Stage 2 — BERTweet encoder   (12 transformer layers → 768-dim [CLS] vector)
#   Stage 3 — Classification head (768-dim → 3-class softmax → stance scalar)
#
# This file defines the PyTorch model class and the tokenizer helper.
# Training and inference are handled separately in stage3_training.py.
# ───────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from config import BERTWEET_MODEL, MAX_LENGTH, NUM_LABELS, HIDDEN_DIM, DROPOUT, POOLING


# ── Stage 1: Tokenizer ─────────────────────────────────────────────────────

def load_tokenizer():
    """
    Load the BERTweet tokenizer.

    The tokenizer uses Byte-Pair Encoding (BPE) to split tweet text into
    subword tokens drawn from a vocabulary of ~64,000 units built during
    pre-training on 850M tweets.

    It also prepends the special [CLS] token to every sequence — this is
    the token whose final 768-dim vector we use as the tweet representation
    in Stage 2.
    """
    tokenizer = AutoTokenizer.from_pretrained(BERTWEET_MODEL, use_fast=False)
    return tokenizer


def tokenize_batch(texts: list[str], tokenizer) -> dict:
    """
    Convert a list of cleaned tweet strings into tensors ready for the encoder.

    Parameters
    ----------
    texts : list of str
        Cleaned tweet strings (output of Stage 0).
    tokenizer :
        The BERTweet tokenizer returned by load_tokenizer().

    Returns
    -------
    dict with keys:
        'input_ids'      : LongTensor [batch, MAX_LENGTH]  — token ID sequences
        'attention_mask' : LongTensor [batch, MAX_LENGTH]  — 1 for real tokens,
                           0 for padding
    """
    encoded = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding="max_length",   # pad shorter tweets to MAX_LENGTH
        truncation=True,        # truncate tweets longer than MAX_LENGTH
        return_tensors="pt",    # return PyTorch tensors
    )
    # we only need input_ids and attention_mask for classification
    return {
        "input_ids":      encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


# ── Stage 2 + 3: BERTweet Encoder + Classification Head ───────────────────

class BERTweetStanceClassifier(nn.Module):
    """
    Full model combining Stages 2 and 3:

    Stage 2 — BERTweet encoder
        12 transformer layers with multi-head self-attention.
        Input  : token IDs + attention mask  [batch, MAX_LENGTH]
        Output : [CLS] token embedding       [batch, 768]

    Stage 3 — Classification head
        A small feed-forward network on top of the frozen/fine-tuned encoder.
        Input  : [CLS] embedding  [batch, 768]
        Output : stance scalar    [batch]   in range [-1, +1]
                 (also exposes raw logits and softmax probs for analysis)
    """

    def __init__(self):
        super().__init__()

        # ── Stage 2: BERTweet encoder ──────────────────────────────────────
        # AutoModel loads the 12-layer transformer without any task-specific
        # head — just the raw encoder that outputs a 768-dim vector per token.
        self.encoder = AutoModel.from_pretrained(BERTWEET_MODEL)

        # ── Stage 3: Classification head ───────────────────────────────────
        # Dropout applied to the pooled tweet embedding before the head, and
        # again between the two linear layers — the extra inner dropout helps
        # regularise the small hand-labelled set against the much larger,
        # noisier weak-label set.
        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Sequential(
            nn.Linear(768, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, NUM_LABELS),  # [Pro-Biden, Pro-Trump, Spam]
        )

        self.softmax = nn.Softmax(dim=-1)

        if POOLING not in ("cls", "mean"):
            raise ValueError(f"POOLING must be 'cls' or 'mean', got {POOLING!r}")
        self._pooling = POOLING

    # ── forward pass ───────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        input_ids      : LongTensor [batch, MAX_LENGTH]
        attention_mask : LongTensor [batch, MAX_LENGTH]

        Returns
        -------
        dict with:
            'logits'    : raw pre-softmax scores      [batch, 3]
            'probs'     : softmax probabilities       [batch, 3]
                          columns: [P(Biden), P(Trump), P(Spam)]
            'stance'    : continuous scalar [-1, +1]  [batch]
                          = P(Trump) - P(Biden)  (spam mass reduces magnitude)
            'prob_spam' : P(Spam)                     [batch]
        """

        # ── Stage 2: encode the tweet ──────────────────────────────────────
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = encoder_output.last_hidden_state  # [batch, seq, 768]

        if self._pooling == "cls":
            pooled = hidden[:, 0, :]               # [batch, 768]
        else:
            # Masked mean pooling: average hidden states across real tokens
            # only, ignoring padding. Typically +1–3 F1 over [CLS] for tweet
            # classification because the signal is spread across the tweet
            # rather than concentrated in a single pretraining-oriented token.
            mask = attention_mask.unsqueeze(-1).float()      # [batch, seq, 1]
            summed = (hidden * mask).sum(dim=1)              # [batch, 768]
            counts = mask.sum(dim=1).clamp(min=1e-9)         # [batch, 1]
            pooled = summed / counts                         # [batch, 768]

        # ── Stage 3: classify ──────────────────────────────────────────────
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)           # [batch, 3]
        probs  = self.softmax(logits)              # [batch, 3]

        # Renormalize stance over just the political classes so spam mass
        # doesn't compress the magnitude. A tweet with (B=0.1, T=0.8, S=0.1)
        # should read stance ≈ 0.78, not 0.7.
        prob_spam = probs[:, 2]                    # [batch]
        political_mass = probs[:, 0] + probs[:, 1] + 1e-9
        stance = (probs[:, 1] - probs[:, 0]) / political_mass

        return {
            "logits":    logits,
            "probs":     probs,
            "stance":    stance,
            "prob_spam": prob_spam,
        }

    # ── convenience: encode a pre-tokenized batch ─────────────────────────

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
    ) -> dict:
        """
        Inference-only forward pass (no gradient tracking).
        Moves tensors to the correct device automatically.
        """
        self.eval()
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        return self.forward(input_ids, attention_mask)
