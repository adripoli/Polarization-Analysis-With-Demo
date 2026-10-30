# adjectives.py
#
# AI-generated-ish lists of clearly positive/negative adjectives. used by
# analyze_tweets.py to catch tweets that have a candidate keyword but are
# actually dunking on them, not supporting them, e.g. "those MAGA people
# are crazy" (keyword MAGA + negative word "crazy" = flip the polarity)
#
# not perfect - words like "crazy" can be a compliment too ("crazy good",
# "crazy talented") so this'll get it wrong sometimes

POSITIVE_ADJECTIVES = [
    "great", "strong", "honest", "brilliant", "incredible", "respected",
    "principled", "steady", "trustworthy", "inspiring", "amazing",
    "fantastic", "smart", "capable", "genuine", "bold",
]

NEGATIVE_ADJECTIVES = [
    "crazy", "incompetent", "corrupt", "pathetic", "disgusting",
    "dangerous", "dishonest", "clueless", "chaotic", "embarrassing",
    "senile", "unhinged", "racist", "delusional", "unfit", "reckless",
]
