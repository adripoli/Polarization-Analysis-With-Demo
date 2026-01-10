# main py Orchestrates

import argparse
import os
import torch

from config import MODEL_DIR, OUTPUT_DIR
from stage0_cleaning         import load_and_clean
from stage3_training         import train, run_inference
from stage4_5_user_analysis  import aggregate_users, compute_polarization
from stage6_analysis         import run_population_analysis
from model                   import BERTweetStanceClassifier


def get_device() -> torch.device:
    """Use GPU if"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU found — using CPU. Training will be slow; "
              "consider Google Colab for the training step.")
    return device


def load_saved_model(device: torch.device) -> BERTweetStanceClassifier:
    """Load a previously"""
    path = os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at {path}. "
            f"Run without --skip-training first."
        )
    model = BERTweetStanceClassifier().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Loaded saved model from {path}")
    return model


def main(skip_training: bool = False, rebuild_cache: bool = False):
    print("=" * 60)
    print("  Political Polarization Pipeline")
    print("=" * 60)

    device = get_device()

    # Stage 0 Clean
    full_df, filtered_df = load_and_clean(use_cache=not rebuild_cache)

    # Stage 3 Train
    if skip_training:
        model = load_saved_model(device)
    else:
        model = train(full_df, device)

    # Stage 3 inference
    scored_df = run_inference(filtered_df, model, device)

    # Stages 4 5
    user_df = aggregate_users(scored_df)
    user_df = compute_polarization(user_df)

    # Save user scores
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    user_scores_path = os.path.join(OUTPUT_DIR, "user_polarization_scores.csv")
    user_df.to_csv(user_scores_path, index=False)
    print(f"\n  User scores saved to {user_scores_path}")

    # Stage 6 Population
    gmm_stats = run_population_analysis(user_df)

    print("\n" + "=" * 60)
    print("  Pipeline complete. Results saved to " + OUTPUT_DIR)
    print("=" * 60)
    print(f"\n  GMM overlap coefficient : {gmm_stats['gmm_overlap']:.4f}")
    print(f"  (closer to 0 = more polarized population)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Political polarization pipeline"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load a saved model from disk instead of fine-tuning",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore the cached Stage 0 output and re-run cleaning + langdetect",
    )
    args = parser.parse_args()
    main(
        skip_training=args.skip_training,
        rebuild_cache=args.rebuild_cache,
    )
