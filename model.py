# model.py
#
# mark 4: giving up on keyword/adjective matching, training an actual
# classifier instead. starting with just the tokenizer side - fine-tuning
# BERTweet (a BERT model pretrained on 850M tweets so it already gets
# hashtags/emojis/slang) on the ~1000 tweets labeled by hand

from transformers import AutoTokenizer

BERTWEET_MODEL = "vinai/bertweet-base"
MAX_LENGTH = 128
NUM_LABELS = 2  # 0 = pro-biden, 1 = pro-trump


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BERTWEET_MODEL)


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
