# config py Central

# Paths
TRUMP_CSV      = "trump_tweets.csv"
BIDEN_CSV      = "biden_tweets.csv"
OUTPUT_DIR     = "outputs/mark7/"              # new folder old
MODEL_DIR      = "model/mark7/"                # same deal for
STAGE0_CACHE   = "outputs/cache/stage0_cleaned.pkl"   # cached clean lang

# Column names expected
COL_USER    = "user_id"
COL_TEXT    = "tweet"
COL_TWEET   = "tweet_id"
COL_TIME    = "created_at"

# Stage 0 Cleaning
MIN_TWEETS_PER_USER = 3       # users with fewer
FILTER_LANGUAGE     = "en"     # ISO 639 1

# Stage 1 2
BERTWEET_MODEL  = "vinai/bertweet-base"
MAX_LENGTH      = 128          # max tokens per
BATCH_SIZE      = 32           # reduce to 16

# Stage 3 Classification
MANUAL_LABELS_CSV = "data/manual_labels.csv"   # set to None
MANUAL_LABEL_WEIGHT = 20        # repeat manual samples
LOCKED_TEST_CSVS = ["data/heldout_test.csv", "data/shortcut_test.csv", "data/fresh_probe.csv"]
WEAK_PER_CELL    = 12000       # tweets per side
KEEP_TAG_SHARE   = 0.25        # leave side tags
NUM_LABELS      = 3            # 0 Pro Biden
HIDDEN_DIM      = 256          # intermediate layer size
DROPOUT         = 0.3
LEARNING_RATE   = 2e-5         # encoder LR pretrained
HEAD_LEARNING_RATE = 1e-4      # classifier head LR
EPOCHS          = 6            # max epochs early
EARLY_STOPPING_PATIENCE = 2    # stop if val
TRAIN_SPLIT     = 0.8          # 80 train 20
LABEL_SMOOTHING = 0.1          # softens targets to
POOLING         = "mean"       # mean masked mean
USE_AMP         = True         # mixed precision training
NUM_WORKERS     = 0            # DataLoader workers bump
INFERENCE_BATCH_SIZE = 64      # larger than train

# Stage 4 Spam
SPAM_USER_THRESHOLD = 0.5      # users with this

# Rule based spam
MIN_CONTENT_WORDS          = 5     # tweets with fewer
MIN_CLASSIFIER_CONFIDENCE  = 0.55  # if max P_Biden

# Stage 5 Polarization
WEIGHT_STD          = 0.6
WEIGHT_ENTROPY      = 0.2
WEIGHT_CONSISTENCY  = 0.2

# Stage 6 Population
GMM_COMPONENTS  = 2            # we expect two
HISTOGRAM_BINS  = 80
