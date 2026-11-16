import pandas as pd
import numpy as np

# Load the exported CSV
df = pd.read_csv("results/per_patient_test_dice.csv")

# Filter for Severity 0 (Baseline) to generate confidence intervals for the main results
df_clean = df[df["severity"] == 0]

# Average across the 3 seeds for each patient to get a stable model score per patient
df_avg = df_clean.groupby(["label_source", "arch", "patient_id"])["test_dice"].mean().reset_index()

# Bootstrap configuration
n_bootstraps = 10000
np.random.seed(42) # For reproducibility

def get_bootstrap_ci(data):
    bootstrapped_means = []
    n = len(data)
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_means.append(np.mean(sample))
    return np.percentile(bootstrapped_means, 2.5), np.percentile(bootstrapped_means, 97.5)

print("=== 95% Bootstrap Confidence Intervals (Severity 0) ===")
for source in ["otsu", "kmeans", "fcm"]:
    print(f"\nLabel Source: {source.upper()}")
    for arch in ["resnet34", "efficientnet-b0", "mit_b0", "tu-mobilevit_s"]:
        # Extract the 10 patient scores for this specific combination
        data = df_avg[(df_avg["label_source"] == source) & (df_avg["arch"] == arch)]["test_dice"].values
        
        if len(data) == 10:
            ci_low, ci_high = get_bootstrap_ci(data)
            mean_val = np.mean(data)
            print(f"{arch:17s}: Mean = {mean_val:.4f} | 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        else:
            print(f"{arch:17s}: Missing data (Found {len(data)} patients)")