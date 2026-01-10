import pandas as pd
import re
import os

def clean_tweet(text):
    """Clean tweet text"""
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_manual_labels():
    """Extract first 500"""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    trump_csv = os.path.join(base_dir, "trump_tweets.csv")
    biden_csv = os.path.join(base_dir, "biden_tweets.csv")
    output_csv = os.path.join(base_dir, "data", "manual_labels.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Load data
    print("Loading tweets...")
    trump_df = pd.read_csv(trump_csv, nrows=500)
    biden_df = pd.read_csv(biden_csv, nrows=500)

    print(f"  Trump tweets loaded: {len(trump_df)}")
    print(f"  Biden tweets loaded: {len(biden_df)}")

    # Create manual labels
    manual_labels = []

    # Trump tweets label
    print("\nProcessing Trump tweets...")
    for idx, row in trump_df.iterrows():
        cleaned_text = clean_tweet(row['tweet'])
        if cleaned_text:  # only add if
            manual_labels.append({
                'cleaned_text': cleaned_text,
                'label': 1
            })

    # Biden tweets label
    print("Processing Biden tweets...")
    for idx, row in biden_df.iterrows():
        cleaned_text = clean_tweet(row['tweet'])
        if cleaned_text:  # only add if
            manual_labels.append({
                'cleaned_text': cleaned_text,
                'label': 0
            })

    result_df = pd.DataFrame(manual_labels)

    # Save to CSV
    result_df.to_csv(output_csv, index=False)

    print(f"\n=== Manual Labels Created ===")
    print(f"Total tweets: {len(result_df)}")
    print(f"  Pro-Biden (0):  {(result_df['label'] == 0).sum()}")
    print(f"  Pro-Trump (1):  {(result_df['label'] == 1).sum()}")
    print(f"\nSaved to: {output_csv}")

if __name__ == "__main__":
    create_manual_labels()
