"""
03.2.1_calculate_sweetspot.py
=====================================

Purpose
-------
Create two kinds of visual summaries to compare two models across multiple
species and metrics, over a range of confidence thresholds:

1) Single-page matrix of boxplots
   - Rows: metrics (Kendall correlation, scaling factor, out/in-season ratio)
   - Columns: species
   - Each subplot overlays boxplots for the two selected models across thresholds.
   Output:
     03_correlation/03.2_Correlation_threshold_sweetspot/03.6_single_page_matrix/model_comparison_matrix.png
     03_correlation/03.2_Correlation_threshold_sweetspot/03.6_single_page_matrix/model_comparison_matrix.pdf

2) Single-page matrix of curves with min–max shading
   - Same grid (rows = metrics, columns = species).
   - For each model, plot the mean of the metric vs transformed threshold,
     with a shaded band showing min–max across validation datasets.
   Output:
     03_correlation/03.2_Correlation_threshold_sweetspot/03.6_single_page_matrix/model_comparison_curves.png
     03_correlation/03.2_Correlation_threshold_sweetspot/03.6_single_page_matrix/model_comparison_curves.pdf

Inputs
------
- CSV produced earlier in the pipeline (step 03.1):
  03_correlation/03.1_tables/table_thresholds_corrK_SF_SigToNoise.csv

  Expected columns include at least:
    species, model, dataset, threshold,
    kendall_corr, scaling_factor, ratio_mean,
    season_start, season_end

Parameters to edit
------------------
- `model_a`, `model_b` to choose the two models to compare.
- `species_list` to select which species to display.
- Output directory at:
  03_correlation/03.2_Correlation_threshold_sweetspot/03.6_single_page_matrix

Dependencies
------------
- pandas
- matplotlib
- seaborn
- numpy

Notes
-----
- Thresholds are shown as-is on the X-axis for the boxplots.
- For the curve plots, the X-axis uses the transform X = -log10(1 - threshold).
- The script filters out rows without a detected/custom season (missing season_start/season_end).
- If your folder layout differs, adjust the `input_csv` path and the output directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch

# Load the table of metrics produced by step 03.1
input_csv = "03_correlation/03.1_tables/table_thresholds_corrK_SF_SigToNoise.csv"
df = pd.read_csv(input_csv)

# Models to compare (edit these labels as needed)
model_a = "2025Q2_Gamma_onnx_mixed_15sp_redFluo"
model_b = "2025Q1_Beta_onnx_mixed_15sp"
models_to_compare = [model_a, model_b]

# Colors for the two models (used in both boxplots and curves)
model_colors = {
    model_a: "#009E73",
    model_b: "#0335fc",
}

# Species and metrics to display
species_list = ["Betula", "Poaceae", "Quercus", "Fagus", "Fraxinus", "Alnus", "Corylus"]
metrics = ["kendall_corr", "scaling_factor", "ratio_mean"]

# Keep only the two chosen models and the requested species; require a season
df = df[df["model"].isin(models_to_compare)]
df = df[df["species"].isin(species_list)]
df = df.dropna(subset=["season_start", "season_end"])

# Where results will be written
output_dir = "03_correlation/03.2_Correlation_threshold_sweetspot/03.6_single_page_matrix"
os.makedirs(output_dir, exist_ok=True)

# Grid size for the single-page matrix of boxplots
n_rows = len(metrics)
n_cols = len(species_list)
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=False)

# Build each subplot: one metric per row, one species per column
for i, metric in enumerate(metrics):
    for j, species in enumerate(species_list):
        ax = axes[i][j]
        df_subset = df[(df["species"] == species)][["threshold", "model", metric]].copy()
        df_subset = df_subset.rename(columns={metric: "Value"})

        if df_subset.empty:
            ax.axis("off")
            continue

        # Overlay boxplots for both models within the same axes
        for model in models_to_compare:
            model_data = df_subset[df_subset["model"] == model]
            sns.boxplot(
                data=model_data,
                x="threshold",
                y="Value",
                linewidth=1,
                fliersize=2,
                width=0.6,
                ax=ax,
                boxprops=dict(facecolor=model_colors[model], edgecolor=model_colors[model], alpha=0.4),
                whiskerprops=dict(color=model_colors[model], alpha=0.3),
                capprops=dict(color=model_colors[model], alpha=0.3),
                medianprops=dict(color=model_colors[model], alpha=1),
                flierprops=dict(marker='o', markersize=3, linestyle='none', markerfacecolor=model_colors[model], alpha=0.4),
                showfliers=False
            )

        # Titles and axis labels
        ax.set_title(f"{species}" if i == 0 else "", fontsize=10)
        if i == n_rows - 1:
            ax.set_xlabel("Threshold", fontsize=9)
        else:
            ax.set_xlabel("")
        if j == 0:
            label = {
                "kendall_corr": "Kendall",
                "scaling_factor": "Scaling",
                "ratio_mean": "Ratio (log)"
            }[metric]
            ax.set_ylabel(label, fontsize=9)
        else:
            ax.set_ylabel("")

        # Reasonable y-axis ranges per metric
        if metric == "kendall_corr":
            ax.set_ylim(0, 1.05)
        elif metric == "scaling_factor":
            ax.set_ylim(0, 45)
        elif metric == "ratio_mean":
            ax.set_yscale("log")
            ax.set_ylim(1e-2, 10)

        ax.tick_params(labelsize=6)
        ax.grid(axis="y", linestyle=":", linewidth=0.5)
        ax.tick_params(axis="x", labelrotation=90)

# Legend showing which color corresponds to which model
legend_elements = [
    Patch(facecolor=model_colors[model], alpha=0.4, label=model)
    for model in models_to_compare
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9, title="Model")

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
fig.suptitle("Comparison of Models per Species and Metric", fontsize=16)

# Save the single-page matrix of boxplots
outpath_base = os.path.join(output_dir, "model_comparison_matrix")
plt.savefig(f"{outpath_base}.png", dpi=300)
plt.savefig(f"{outpath_base}.pdf")
plt.close()
print(f"Saved combined matrix plot at: {outpath_base}.png/.pdf")

############################################### curves

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

# Load the same input table again (kept separate from above for clarity)
input_csv = "03_correlation/03.1_tables/table_thresholds_corrK_SF_SigToNoise.csv"
df = pd.read_csv(input_csv)

# Parameters: the same two models and lists as above
model_a = "2025Q2_Gamma_onnx_mixed_15sp_redFluo"
model_b = "2025Q1_Beta_onnx_mixed_15sp"
models_to_compare = [model_a, model_b]

model_colors = {
    model_a: "#009E73",
    model_b: "#0335fc",
}

species_list = ["Betula", "Poaceae", "Quercus", "Fagus", "Fraxinus", "Alnus", "Corylus"]
metrics = ["kendall_corr", "scaling_factor", "ratio_mean"]

# Keep only the two models, requested species, and rows that have seasons
df = df[df["model"].isin(models_to_compare)]
df = df[df["species"].isin(species_list)]
df = df.dropna(subset=["season_start", "season_end"])
df = df[df["threshold"] > 0]  # Avoid log transform issues at 0

# Output folder for the curves with min–max shading
output_dir = "03_correlation/03.2_Correlation_threshold_sweetspot/03.6_single_page_matrix"
os.makedirs(output_dir, exist_ok=True)

# Set up the grid for the curve plots
n_rows = len(metrics)
n_cols = len(species_list)
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=False)

legend_elements = [
    Patch(facecolor=model_colors[model], alpha=0.4, label=model)
    for model in models_to_compare
]

# Labels to hide on the X-axis for readability
hidden_thresholds = {0.4, 0.75, 0.85, 0.91, 0.92, 0.93, 0.94, 0.96, 0.97, 0.98}
thresholds_sorted = sorted(df["threshold"].unique())
log_thresholds = np.log10(thresholds_sorted)
ax.set_xticks(log_thresholds)

# Build each subplot: one metric per row, one species per column
for i, metric in enumerate(metrics):
    for j, species in enumerate(species_list):
        ax = axes[i][j]
        df_subset = df[df["species"] == species][["threshold", "model", metric]].copy()
        df_subset = df_subset.rename(columns={metric: "Value"})

        if df_subset.empty:
            ax.axis("off")
            continue

        # For each model, compute per-threshold mean, min, and max across datasets
        for model in models_to_compare:
            model_data = df_subset[df_subset["model"] == model]
            if model_data.empty:
                continue

            grouped = model_data.groupby("threshold")["Value"].agg(["mean", "min", "max"]).reset_index()
            grouped["log_threshold"] = -np.log10(1 - grouped["threshold"].clip(upper=0.9999999))

            ax.plot(grouped["log_threshold"], grouped["mean"], label=model, color=model_colors[model], linewidth=1.5)
            ax.fill_between(grouped["log_threshold"], grouped["min"], grouped["max"], color=model_colors[model], alpha=0.3)

        # Titles and axis labels
        ax.set_title(f"{species}" if i == 0 else "", fontsize=10)
        if i == n_rows - 1:
            ax.set_xlabel("-log(1 - threshold)", fontsize=9)
        else:
            ax.set_xlabel("")
        if j == 0:
            label = {
                "kendall_corr": "Kendall",
                "scaling_factor": "Scaling",
                "ratio_mean": "Ratio (log)"
            }[metric]
            ax.set_ylabel(label, fontsize=9)
        else:
            ax.set_ylabel("")

        # Reasonable y-axis ranges per metric
        if metric == "kendall_corr":
            ax.set_ylim(0, 1.05)
        elif metric == "scaling_factor":
            ax.set_ylim(0, 45)
        elif metric == "ratio_mean":
            ax.set_yscale("log")
            ax.set_ylim(1e-2, 10)

        # X-ticks based on the same transform, with some labels hidden
        xticks_raw = np.sort(df["threshold"].unique())
        xticks_log = -np.log10(1 - np.clip(xticks_raw, None, 0.9999999))
        ax.set_xticks(xticks_log)
        ax.set_xticklabels([f"{t:.2g}" for t in xticks_raw], rotation=90, fontsize=6)
        xtick_labels = [f"{t:.2g}" if t not in hidden_thresholds else "" for t in thresholds_sorted]
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)

        ax.tick_params(labelsize=6)
        ax.grid(axis="both", linestyle=":", linewidth=0.5)

# Final title and legend, then save
fig.suptitle("Model Comparison with Spread (Min–Max)", fontsize=16)
fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9, title="Model")
plt.tight_layout(rect=[0, 0.05, 1, 0.97])

output_base = os.path.join(output_dir, "model_comparison_curves")
plt.savefig(f"{output_base}.png", dpi=300)
plt.savefig(f"{output_base}.pdf")
plt.close()

print(f"Saved plots at: {output_base}.png / .pdf")
