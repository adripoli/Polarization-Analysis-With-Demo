# model.py
#
# mark 4: giving up on keyword/adjective matching, training an actual
# classifier instead. fine-tuning BERTweet (a BERT model pretrained on 850M
# tweets so it already gets hashtags/emojis/slang) with a small head on top,
# using ~1000 tweets i labeled by hand as pro-biden (0) or pro-trump (1)

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from config import MAX_LENGTH

BERTWEET_MODEL = "vinai/bertweet-base"
NUM_LABELS = 2  # 0 = pro-biden, 1 = pro-trump


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BERTWEET_MODEL, use_fast=False)


def tokenize_batch(texts, tokenizer):
    encoded = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


class BERTweetStanceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BERTWEET_MODEL)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_LABELS),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)

        logits = self.classifier(cls_embedding)
        probs = self.softmax(logits)

        # -1 = full biden, +1 = full trump
        stance = probs[:, 1] - probs[:, 0]

        return {"logits": logits, "probs": probs, "stance": stance}
