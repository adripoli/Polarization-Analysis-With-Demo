# bertweet_spike py throwaway

from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "vinai/bertweet-base"

SAMPLE_TWEETS = [
    "MAGA all the way!! #Trump2020 is going to win this thing 🇺🇸🔥",
    "lmaooo bidenharris2020 got my vote, dude's actually not that bad ngl",
    "can't believe people still fall for this clown lol #drainswamp",
]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    for tweet in SAMPLE_TWEETS:
        print(tweet)
        print(tokenizer.tokenize(tweet))
        print()

    # keeps hashtags slang
