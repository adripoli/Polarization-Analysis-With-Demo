# stage6_analysis py stage

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import pearsonr

from config import (
    OUTPUT_DIR, HISTOGRAM_BINS, GMM_COMPONENTS,
)


# helpers

def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1 stance histogram

def plot_stance_histogram(user_df: pd.DataFrame) -> None:
    """plots the distribution"""
    _ensure_output_dir()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        user_df["mean_stance"],
        bins=HISTOGRAM_BINS,
        color="#4a90d9",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
    )

    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Center (0)")
    ax.set_xlabel("Mean Stance Score  (−1 = Pro-Biden, +1 = Pro-Trump)", fontsize=12)
    ax.set_ylabel("Number of Users", fontsize=12)
    ax.set_title("Distribution of User Mean Stance Scores", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "stance_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# 2 gaussian mixture

def fit_gmm(user_df: pd.DataFrame) -> dict:
    """fits a GMM"""
    _ensure_output_dir()

    X = user_df["mean_stance"].values.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=GMM_COMPONENTS,
        covariance_type="full",
        random_state=42,
        max_iter=200,
    )
    gmm.fit(X)

    means = gmm.means_.flatten()
    stds  = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    # sort components left
    order  = np.argsort(means)
    means  = means[order]
    stds   = stds[order]
    weights = weights[order]

    # overlap coefficient numerically
    from scipy.stats import norm as scipy_norm
    x_range = np.linspace(-1.5, 1.5, 2000)
    pdf0 = weights[0] * scipy_norm.pdf(x_range, means[0], stds[0])
    pdf1 = weights[1] * scipy_norm.pdf(x_range, means[1], stds[1])
    overlap = np.trapezoid(np.minimum(pdf0, pdf1), x_range)

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        X.flatten(),
        bins=HISTOGRAM_BINS,
        density=True,
        color="#ccddee",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.7,
        label="Observed distribution",
    )

    colors = ["#e74c3c", "#2980b9"]
    labels = ["Left cluster (Pro-Biden)", "Right cluster (Pro-Trump)"]
    for i in range(GMM_COMPONENTS):
        ax.plot(
            x_range,
            weights[i] * scipy_norm.pdf(x_range, means[i], stds[i]),
            color=colors[i],
            linewidth=2,
            label=f"{labels[i]}: μ={means[i]:.2f}, σ={stds[i]:.2f}",
        )

    ax.fill_between(
        x_range,
        np.minimum(pdf0, pdf1),
        alpha=0.3,
        color="grey",
        label=f"Overlap ≈ {overlap:.3f}",
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Mean Stance Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Gaussian Mixture Model Fit to User Stance Distribution",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gmm_fit.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    stats = {
        "gmm_mean_left":    float(means[0]),
        "gmm_mean_right":   float(means[1]),
        "gmm_std_left":     float(stds[0]),
        "gmm_std_right":    float(stds[1]),
        "gmm_weight_left":  float(weights[0]),
        "gmm_weight_right": float(weights[1]),
        "gmm_overlap":      float(overlap),
    }

    print(f"\n  GMM results:")
    for k, v in stats.items():
        print(f"    {k:<25} {v:.4f}")

    return stats


# 3 polarization vs

def plot_polarization_vs_activity(user_df: pd.DataFrame) -> None:
    """scatter of polarization"""
    _ensure_output_dir()

    r, p = pearsonr(user_df["tweet_count"], user_df["polarization_weighted"])

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        user_df["tweet_count"],
        user_df["polarization_weighted"],
        alpha=0.15,
        s=6,
        color="#4a90d9",
    )

    # trend line
    z = np.polyfit(user_df["tweet_count"], user_df["polarization_weighted"], 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(user_df["tweet_count"].min(),
                         user_df["tweet_count"].max(), 200)
    ax.plot(x_line, p_fit(x_line), color="red", linewidth=1.5,
            label=f"Trend  (r = {r:.3f}, p = {p:.3f})")

    ax.set_xlabel("Tweet Count per User", fontsize=12)
    ax.set_ylabel("Polarization Score", fontsize=12)
    ax.set_title("Polarization Score vs. User Activity",
                 fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "polarization_vs_activity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Pearson r = {r:.3f}  (p = {p:.4f})")


# 4 save summary

def save_results(user_df: pd.DataFrame, gmm_stats: dict) -> None:
    """saves the full"""
    _ensure_output_dir()

    user_path = os.path.join(OUTPUT_DIR, "user_polarization_scores.csv")
    user_df.to_csv(user_path, index=False)
    print(f"  Saved: {user_path}")

    gmm_path = os.path.join(OUTPUT_DIR, "gmm_summary.csv")
    pd.DataFrame([gmm_stats]).to_csv(gmm_path, index=False)
    print(f"  Saved: {gmm_path}")


# main entry point

def run_population_analysis(user_df: pd.DataFrame) -> dict:
    """runs all the"""
    print("\n=== Stage 6: Population-Level Analysis ===")

    plot_stance_histogram(user_df)
    gmm_stats = fit_gmm(user_df)
    plot_polarization_vs_activity(user_df)
    save_results(user_df, gmm_stats)

    return gmm_stats
