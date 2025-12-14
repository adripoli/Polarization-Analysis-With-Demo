# config py constants

TRUMP_CSV = "trump_tweets.csv"
BIDEN_CSV = "biden_tweets.csv"
MANUAL_LABELS_CSV = "data/manual_labels.csv"

MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_SPLIT = 0.8

NUM_LABELS = 3  # 0 pro biden

# users with more
SPAM_USER_THRESHOLD = 0.5

# weights for the
WEIGHT_STD = 0.6
WEIGHT_ENTROPY = 0.2
WEIGHT_CONSISTENCY = 0.2

MIN_WORDS = 5

# users below this
MIN_TWEETS_PER_USER = 3
