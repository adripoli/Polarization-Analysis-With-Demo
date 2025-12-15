# model.py
#
# stages 1-3 (the actual model):
#   stage 1 - BERTweet tokenizer, turns cleaned text into token id tensors
#   stage 2 - BERTweet encoder, 12 transformer layers -> 768-dim [CLS] vector
#   stage 3 - classification head, 768-dim -> 3-class softmax -> stance number
#
# this file just has the model class + tokenizer helper. training/inference
# live in stage3_training.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from config import BERTWEET_MODEL, MAX_LENGTH, NUM_LABELS, HIDDEN_DIM, DROPOUT


# stage 1: tokenizer

def load_tokenizer():
    """loads the BERTweet tokenizer.

    uses BPE to split tweets into subword tokens from a ~64k vocab built
    during pretraining on 850M tweets. also sticks a [CLS] token on the
    front of every sequence - that's the one whose 768-dim output we use
    as the tweet's representation in stage 2."""
    tokenizer = AutoTokenizer.from_pretrained(BERTWEET_MODEL, use_fast=False)
    return tokenizer


def tokenize_batch(texts: list[str], tokenizer) -> dict:
    """turns a list of cleaned tweets into tensors the encoder can eat.

    takes: list of cleaned tweet strings (stage 0 output), the tokenizer
    returns: dict with input_ids [batch, MAX_LENGTH] and attention_mask
    (1 = real token, 0 = padding)"""
    encoded = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding="max_length",   # pad short tweets up to MAX_LENGTH
        truncation=True,        # chop anything longer
        return_tensors="pt",    # give us pytorch tensors
    )
    # only need these two for classification
    return {
        "input_ids":      encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


# stage 2 + 3: BERTweet encoder + classification head

class BERTweetStanceClassifier(nn.Module):
    """combines stages 2 and 3:

    stage 2 (encoder): 12 transformer layers w/ self-attention.
        in: token ids + mask [batch, MAX_LENGTH] -> out: [CLS] vector [batch, 768]

    stage 3 (head): small feedforward net on top.
        in: [CLS] embedding [batch, 768] -> out: stance number [batch] in [-1, +1]
        (also gives you raw logits + softmax probs if you need them)"""

    def __init__(self):
        super().__init__()

        # stage 2: BERTweet encoder - just the raw 12-layer transformer,
        # no task-specific head, outputs a 768-dim vector per token
        self.encoder = AutoModel.from_pretrained(BERTWEET_MODEL)

        # stage 3: classification head
        # dropout zeroes 30% of activations during training so it doesn't
        # overfit on the (pretty small) labeled set
        self.dropout = nn.Dropout(DROPOUT)

        # two linear layers w/ ReLU between them: 768 -> HIDDEN_DIM (256) -> NUM_LABELS
        self.classifier = nn.Sequential(
            nn.Linear(768, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_LABELS),  # [pro-biden, pro-trump, spam]
        )

        self.softmax = nn.Softmax(dim=-1)

    # forward pass

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """returns a dict with:
            logits    : raw scores before softmax [batch, 3]
            probs     : softmax probs [batch, 3] -> [P(biden), P(trump), P(spam)]
            stance    : number in [-1, +1] [batch] = P(trump) - P(biden), spam
                        mass pulls the magnitude down
            prob_spam : P(spam) [batch]"""

        # stage 2: encode the tweet
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]

        # stage 3: classify
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)    # [batch, 3]
        probs  = self.softmax(logits)              # [batch, 3]

        # renormalize stance over just the political classes so spam mass
        # doesn't shrink the magnitude. a tweet at (B=0.1, T=0.8, S=0.1)
        # should read stance ~0.78, not 0.7
        prob_spam = probs[:, 2]                    # [batch]
        political_mass = probs[:, 0] + probs[:, 1] + 1e-9
        stance = (probs[:, 1] - probs[:, 0]) / political_mass

        return {
            "logits":    logits,
            "probs":     probs,
            "stance":    stance,
            "prob_spam": prob_spam,
        }

    # convenience: run a pre-tokenized batch through the model

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
    ) -> dict:
        """inference-only forward pass, no gradients. moves tensors to the
        right device for you"""
        self.eval()
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        return self.forward(input_ids, attention_mask)
