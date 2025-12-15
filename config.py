# config py all

# paths
TRUMP_CSV      = "trump_tweets.csv"
BIDEN_CSV      = "biden_tweets.csv"
OUTPUT_DIR     = "outputs/"
MODEL_DIR      = "model/bertweet_finetuned/"   # saved after fine

# column names expected
COL_USER    = "user_id"
COL_TEXT    = "tweet"
COL_TWEET   = "tweet_id"
COL_TIME    = "created_at"

# stage 0 cleaning
MIN_TWEETS_PER_USER = 3       # users below this

# stage 1 2
BERTWEET_MODEL  = "vinai/bertweet-base"
MAX_LENGTH      = 128          # max tokens per
BATCH_SIZE      = 32           # drop to 16

# stage 3 classification
MANUAL_LABELS_CSV = "data/manual_labels.csv"   # set to None
MANUAL_LABEL_WEIGHT = 5        # repeat manual samples
NUM_LABELS      = 3            # 0 pro biden
HIDDEN_DIM      = 256          # size of the
DROPOUT         = 0.3
LEARNING_RATE   = 2e-5
EPOCHS          = 3
TRAIN_SPLIT     = 0.8          # 80 train 20

# stage 4 spam
SPAM_USER_THRESHOLD = 0.5      # users with more

# stage 5 polarization
WEIGHT_STD          = 0.6
WEIGHT_ENTROPY      = 0.2
WEIGHT_CONSISTENCY  = 0.2

# stage 6 population
GMM_COMPONENTS  = 2            # expecting two clusters
HISTOGRAM_BINS  = 80
