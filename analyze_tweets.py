import pandas as pd

TRUMP_CSV = "trump_tweets.csv"
BIDEN_CSV = "biden_tweets.csv"


def main():
    trump_df = pd.read_csv(TRUMP_CSV)
    biden_df = pd.read_csv(BIDEN_CSV)
    df = pd.concat([trump_df, biden_df], ignore_index=True)

    print(f"Loaded {len(df)} tweets")


if __name__ == "__main__":
    main()
