# model.py
#
# mark 4: giving up on keyword/adjective matching, training an actual
# classifier instead. fine-tuning BERTweet (a BERT model pretrained on 850M
# tweets so it already gets hashtags/emojis/slang) with a small head on top,
# using ~1000 tweets i labeled by hand as pro-biden (0) or pro-trump (1)

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from config import MAX_LENGTH, NUM_LABELS

BERTWEET_MODEL = "vinai/bertweet-base"


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
        probs = self.softmax(logits)  # [P(biden), P(trump), P(spam)]

        # naive stance = probs[:, 1] - probs[:, 0] looked right until spam
        # tweets showed up in testing: something like (biden=0.1, trump=0.6,
        # spam=0.3) - clearly pro-trump, spam-flavored phrasing - only
        # scored stance=0.5 instead of something close to 1. the spam mass
        # was eating into the political signal for no good reason, since
        # spam isn't "a bit biden and a bit trump", it's just noise.
        #
        # renormalizing over P(biden)+P(trump) only fixes that: the same
        # tweet now scores (0.6-0.1)/(0.6+0.1) = 0.71, and the score no
        # longer shrinks just because the model also picked up some spam
        # probability mass.
        prob_spam = probs[:, 2]
        political_mass = probs[:, 0] + probs[:, 1] + 1e-9
        stance = (probs[:, 1] - probs[:, 0]) / political_mass

        return {
            "logits": logits,
            "probs": probs,
            "stance": stance,
            "prob_spam": prob_spam,
        }
