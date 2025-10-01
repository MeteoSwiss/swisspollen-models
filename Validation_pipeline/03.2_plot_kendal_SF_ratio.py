"""
03.2_plot_kendal_SF_ratio.py
=================================================

Purpose
-------
Visualize, for multiple confidence thresholds, the Kendall correlation,
scaling factor, and out-of-season vs in-season mean ratio derived from the
table produced in step 03.1 (table_thresholds_corrK_SF_SigToNoise.csv).

What this script produces
-------------------------
1) One PDF per model:
   - A grid of small plots (rows = species, columns = validation datasets).
   - Each small plot shows three curves across thresholds:
       * Kendall's tau correlation (left Y axis)
       * Scaling factor (right Y axis)
       * Ratio of mean predicted counts out-of-season / in-season (second right Y axis, log scale)
   Output folder:
     03_correlation/03.2_Correlation_threshold_sweetspot/03.3.2_kendal_sf_ratio

2) One combined PDF and multiple PNGs:
   - For each species, a single page that contains one panel per model.
   - In each panel, all validation datasets are plotted together with distinct linestyles.
   Output folder:
     03_correlation/03.2_Correlation_threshold_sweetspot/03.3.4_combined_species_per_species

3) Min/Max envelope plots:
   - For each species and model, the min/max band (across datasets) for each metric vs threshold,
     along with the midline of the band.
   - Two variants are produced:
       * All models
         03_correlation/03.2_Correlation_threshold_sweetspot/03.3.6_minmax_envelope
       * Selected models only
         03_correlation/03.2_Correlation_threshold_sweetspot/03.3.6_minmax_envelope_selected

Inputs
------
- CSV file (edit `input_csv`) created by step 03.1:
  03_correlation/03.1_tables/table_thresholds_corrK_SF_SigToNoise.csv

  Expected columns include at least:
    species, model, dataset, threshold,
    kendall_corr, scaling_factor, ratio_mean,
    season_start, season_end

- `target_species`: list of species to plot (edit if needed).
- `selected_models`: list of models for the final “selected models” envelope plots.

Dependencies
------------
- pandas
- numpy
- matplotlib

Notes
-----
- The X-axis uses the transform: X = -log10(1 - threshold).
- To keep labels readable, some thresholds are hidden on the X-axis (see `thresholds_to_hide`).
- The script creates output directories if they do not exist.
- If you are not working under the same folder structure, adjust `input_csv`
  and the output directories at the top of the file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

# Load input table produced by 03.1
input_csv = "03_correlation/03.1_tables/table_thresholds_corrK_SF_SigToNoise.csv"
df = pd.read_csv(input_csv)

# Species to include (edit as needed)
target_species = ["Betula", "Poaceae", "Quercus", "Fagus", "Fraxinus", "Alnus", "Corylus"]

# All models present in the table
models = sorted(df["model"].unique())

# Hide selected thresholds on X-axis to improve readability
thresholds_to_hide = [0.91, 0.92, 0.93, 0.94, 0.96, 0.97, 0.98]

# Colors for the three metrics
colors = {
    "kendall": "#FC6C85",  # pink
    "ratio": "#E69F00",    # orange
    "scaling": "#009E73"   # green-teal
}

# Linestyles used to distinguish datasets
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5))]

# Output folders
output_dir_main = "03_correlation/03.2_Correlation_threshold_sweetspot/03.3.2_kendal_sf_ratio"
output_dir_combined_species = "03_correlation/03.2_Correlation_threshold_sweetspot/03.3.4_combined_species_per_species"

os.makedirs(output_dir_main, exist_ok=True)
os.makedirs(output_dir_combined_species, exist_ok=True)

# Keep only selected species
df = df[df["species"].isin(target_species)]

# Remove combinations without a detected or custom season
df = df.dropna(subset=["season_start", "season_end"])

species_list = sorted(df["species"].unique())
datasets = sorted(df["dataset"].unique())

# =============================== PART 1: Grid per model ===============================

for model in models:
    model_df = df[df["model"] == model]
    if model_df.empty:
        continue

    nrows = len(species_list)
    ncols = len(datasets)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for i, species in enumerate(species_list):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            subset = model_df[(model_df["species"] == species) & (model_df["dataset"] == dataset)]

            if subset.empty:
                ax.axis("off")
                continue

            subset = subset.sort_values("threshold")
            transformed_x = -np.log10(1 - subset["threshold"].clip(upper=0.9999999))

            ax.set_title(f"{species}\n{dataset}", fontsize=8)

            # Kendall (left axis)
            ax.plot(transformed_x, subset["kendall_corr"], color=colors["kendall"], linestyle="-", linewidth=1.5)
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis='y', labelcolor=colors["kendall"], labelsize=6)

            if j == 0:
                ax.set_ylabel("Kendall", color=colors["kendall"], fontsize=6)
            else:
                ax.set_yticklabels([])

            # X-axis labeling for thresholds (some hidden for readability)
            ax.set_xlabel("-log(1-threshold)", fontsize=6)
            ax.set_xticks(transformed_x)
            ax.set_xticklabels(
                [str(t) if t not in thresholds_to_hide else "" for t in subset["threshold"]],
                rotation=90,
                fontsize=5
            )

            # Scaling factor (first right axis)
            ax2 = ax.twinx()
            ax2.plot(transformed_x, np.clip(subset["scaling_factor"], 0, 40), color=colors["scaling"], linestyle=":", linewidth=1.5)
            ax2.set_ylim(0, 40)
            ax2.tick_params(axis='y', labelcolor=colors["scaling"], labelsize=6)

            if j == 0:
                ax2.set_ylabel("Scaling Factor", color=colors["scaling"], fontsize=6)

            # Out/In season mean ratio (second right axis, log scale)
            ax3 = ax.twinx()
            ax3.spines.right.set_position(("axes", 1.15))
            ax3.plot(transformed_x, subset["ratio_mean"], color=colors["ratio"], linestyle="--", linewidth=1.5)
            ax3.set_yscale('log')
            ax3.set_ylim(1e-2, 10)
            ax3.tick_params(axis='y', labelcolor=colors["ratio"], labelsize=6)

            if j == 0:
                ax3.set_ylabel("Mean Out/In Season Ratio (log)", color=colors["ratio"], fontsize=6)

            ax.grid(True, linestyle=":", linewidth=0.5)

    # Common legend
    handles = [
        Line2D([], [], color=colors["kendall"], linestyle="-", label="Kendall Correlation"),
        Line2D([], [], color=colors["scaling"], linestyle=":", label="Scaling Factor"),
        Line2D([], [], color=colors["ratio"], linestyle="--", label="Mean Out/In Season Ratio"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    plt.suptitle(f"Model: {model}", fontsize=14)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    output_pdf = os.path.join(output_dir_main, f"{model}_threshold_analysis.pdf")
    plt.savefig(output_pdf)
    plt.close()
    print(f"Saved grid PDF for model: {model} -> {output_pdf}")

# ================== PART 2: Combined pages per species (models side by side) ==================

output_combined_pdf = os.path.join(output_dir_combined_species, "all_species_combined.pdf")

# Consistent linestyle per dataset across figures
all_datasets = sorted(df["dataset"].unique())
dataset_linestyle_map = {
    dataset: linestyles[i % len(linestyles)]
    for i, dataset in enumerate(all_datasets)
}

with PdfPages(output_combined_pdf) as pdf:
    for species in species_list:
        n_models = len(models)
        ncols = n_models
        nrows = 1

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5), squeeze=False)

        for idx, model in enumerate(models):
            ax1 = axes[0][idx]
            ax2 = ax1.twinx()
            ax3 = ax1.twinx()
            ax3.spines.right.set_position(("axes", 1.2))

            # Filter species/model
            species_model_df = df[(df["model"] == model) & (df["species"] == species)]

            if species_model_df.empty:
                ax1.axis("off")
                ax2.axis("off")
                ax3.axis("off")
                continue

            datasets_for_model = sorted(species_model_df["dataset"].unique())

            for dataset in datasets_for_model:
                subset = species_model_df[species_model_df["dataset"] == dataset].sort_values("threshold")
                if subset.empty:
                    continue

                transformed_x = -np.log10(1 - subset["threshold"].clip(upper=0.9999999))
                linestyle = dataset_linestyle_map[dataset]

                ax1.plot(transformed_x, subset["kendall_corr"], color=colors["kendall"], linestyle=linestyle, linewidth=1.5)
                ax2.plot(transformed_x, np.clip(subset["scaling_factor"], 0, 40), color=colors["scaling"], linestyle=linestyle, linewidth=1.5)
                ax3.plot(transformed_x, subset["ratio_mean"], color=colors["ratio"], linestyle=linestyle, linewidth=1.5)

            ax1.set_title(f"{model}", fontsize=10)
            ax1.grid(True, linestyle=":", linewidth=0.5)

            # Y labels only on first column for readability
            if idx == 0:
                ax1.set_ylabel("Kendall", fontsize=8, color=colors["kendall"])
                ax2.set_ylabel("Scaling", fontsize=8, color=colors["scaling"])
                ax3.set_ylabel("Ratio (log)", fontsize=8, color=colors["ratio"])
            else:
                ax1.set_yticklabels([])
                ax2.set_yticklabels([])
                ax3.set_yticklabels([])

            ax1.tick_params(axis='y', labelcolor=colors["kendall"], labelsize=6)
            ax2.tick_params(axis='y', labelcolor=colors["scaling"], labelsize=6)
            ax3.tick_params(axis='y', labelcolor=colors["ratio"], labelsize=6)

            ax1.set_ylim(0, 1.05)
            ax2.set_ylim(0, 40)
            ax3.set_yscale('log')
            ax3.set_ylim(1e-4, 10)

            # Show thresholds on X-axis (some labels hidden)
            all_thresholds = species_model_df["threshold"].unique()
            transformed_x = -np.log10(1 - np.clip(all_thresholds, None, 0.9999999))
            ax1.set_xlabel("-log(1-threshold)", fontsize=8)
            ax1.set_xticks(transformed_x)
            ax1.set_xticklabels(
                [str(t) if t not in thresholds_to_hide else "" for t in all_thresholds],
                rotation=90,
                fontsize=6
            )

        # Legend: three metrics + one entry per dataset linestyle
        legend_elements = [
            Line2D([], [], color=colors["kendall"], linestyle='-', label="Kendall"),
            Line2D([], [], color=colors["scaling"], linestyle='-', label="Scaling Factor"),
            Line2D([], [], color=colors["ratio"], linestyle='-', label="Mean Out/In Season Ratio"),
        ] + [
            Line2D([], [], color="black", linestyle=dataset_linestyle_map[ds], label=ds)
            for ds in all_datasets
        ]

        fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=7)
        fig.suptitle(f"Species: {species}", fontsize=16)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        # Save to the multi-page PDF
        pdf.savefig(fig)

        # Also save per-species PNG
        output_png = os.path.join(output_dir_combined_species, f"{species}_combined.png")
        fig.savefig(output_png, dpi=300)
        print(f"Saved combined plots for species {species}: PNG and PDF page")

        plt.close(fig)

print(f"Combined species PDF created: {output_combined_pdf}")

# ============================= Min/Max envelopes across datasets (all models) =============================

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

output_minmax_dir = "03_correlation/03.2_Correlation_threshold_sweetspot/03.3.6_minmax_envelope"
os.makedirs(output_minmax_dir, exist_ok=True)

with PdfPages(os.path.join(output_minmax_dir, "species_minmax_metrics.pdf")) as pdf:
    for species in species_list:
        fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(5 * len(models), 4), squeeze=False)
        fig.suptitle(f"Species: {species}", fontsize=14)

        for col, model in enumerate(models):
            ax1 = axes[0][col]
            ax2 = ax1.twinx()
            ax3 = ax1.twinx()
            ax3.spines.right.set_position(("axes", 1.2))

            subset = df[(df["species"] == species) & (df["model"] == model)]

            if subset.empty:
                ax1.set_title(f"{model} (no data)", fontsize=10)
                ax1.axis("off")
                ax2.axis("off")
                ax3.axis("off")
                continue

            grouped = subset.groupby("threshold")
            thresholds = sorted(grouped.groups.keys())
            transformed_x = -np.log10(1 - np.clip(thresholds, None, 0.9999999))

            kendall_min = []
            kendall_max = []
            scaling_min = []
            scaling_max = []
            ratio_min = []
            ratio_max = []

            for t in thresholds:
                g = grouped.get_group(t)
                kendall_min.append(g["kendall_corr"].min())
                kendall_max.append(g["kendall_corr"].max())
                scaling_min.append(g["scaling_factor"].clip(0, 40).min())
                scaling_max.append(g["scaling_factor"].clip(0, 40).max())
                ratio_min.append(g["ratio_mean"].min())
                ratio_max.append(g["ratio_mean"].max())

            # Kendall
            ax1.fill_between(transformed_x, kendall_min, kendall_max, color=colors["kendall"], alpha=0.3)
            ax1.plot(transformed_x, [(lo + hi) / 2 for lo, hi in zip(kendall_min, kendall_max)], color=colors["kendall"], linewidth=2)
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel("Kendall", color=colors["kendall"])
            ax1.tick_params(axis='y', labelcolor=colors["kendall"], labelsize=6)

            # Scaling
            ax2.fill_between(transformed_x, scaling_min, scaling_max, color=colors["scaling"], alpha=0.3)
            ax2.plot(transformed_x, [(lo + hi) / 2 for lo, hi in zip(scaling_min, scaling_max)], color=colors["scaling"], linewidth=2, linestyle=":")
            ax2.set_ylim(0, 40)
            ax2.set_ylabel("Scaling", color=colors["scaling"])
            ax2.tick_params(axis='y', labelcolor=colors["scaling"], labelsize=6)

            # Ratio
            ax3.fill_between(transformed_x, ratio_min, ratio_max, color=colors["ratio"], alpha=0.3)
            ax3.plot(transformed_x, [(lo + hi) / 2 for lo, hi in zip(ratio_min, ratio_max)], color=colors["ratio"], linewidth=2, linestyle="--")
            ax3.set_yscale('log')
            ax3.set_ylim(1e-2, 10)
            ax3.set_ylabel("Out/In Ratio", color=colors["ratio"])
            ax3.tick_params(axis='y', labelcolor=colors["ratio"], labelsize=6)

            ax1.set_title(f"{model}", fontsize=10)
            ax1.set_xlabel("-log(1 - threshold)", fontsize=8)
            ax1.set_xticks(transformed_x)
            thresholds_to_hide_labels = {"0.91", "0.92", "0.93", "0.94", "0.95", "0.96", "0.97"}
            ax1.set_xticklabels(
                [str(t) if str(t) not in thresholds_to_hide_labels else "" for t in thresholds],
                rotation=90,
                fontsize=6
            )
            ax1.grid(True, linestyle=":", linewidth=0.5)

        # Legend for the envelopes
        handles = [
            Line2D([], [], color=colors["kendall"], linestyle="-", label="Kendall"),
            Line2D([], [], color=colors["scaling"], linestyle=":", label="Scaling Factor"),
            Line2D([], [], color=colors["ratio"], linestyle="--", label="Out/In Season Ratio"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        pdf.savefig(fig)
        print(f"Saved envelope plot for species: {species}")
        plt.close(fig)

print(f"Final PDF saved to: {output_minmax_dir}/species_minmax_metrics.pdf")

# ============================= Envelopes for selected models only =============================

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os

# Choose which models to include in the reduced-width envelope plots
selected_models = [
    "2025Q2_Gamma_onnx_mixed_15sp_redFluo"
]

output_minmax_dir = "03_correlation/03.2_Correlation_threshold_sweetspot/03.3.6_minmax_envelope_selected"
os.makedirs(output_minmax_dir, exist_ok=True)

# Filter the table to the selected models
filtered_df = df[df["model"].isin(selected_models)]
filtered_models = sorted(filtered_df["model"].unique())
filtered_species = sorted(filtered_df["species"].unique())

with PdfPages(os.path.join(output_minmax_dir, "species_minmax_metrics_selected_models.pdf")) as pdf:
    for species in filtered_species:
        fig, axes = plt.subplots(nrows=1, ncols=len(filtered_models), figsize=(5 * len(filtered_models), 4), squeeze=False)
        fig.suptitle(f"Species: {species}", fontsize=14)

        for col, model in enumerate(filtered_models):
            ax1 = axes[0][col]
            ax2 = ax1.twinx()
            ax3 = ax1.twinx()
            ax3.spines.right.set_position(("axes", 1.2))

            subset = filtered_df[(filtered_df["species"] == species) & (filtered_df["model"] == model)]
            if subset.empty:
                ax1.set_title(f"{model} (no data)", fontsize=10)
                ax1.axis("off")
                ax2.axis("off")
                ax3.axis("off")
                continue

            grouped = subset.groupby("threshold")
            thresholds = sorted(grouped.groups.keys())
            transformed_x = -np.log10(1 - np.clip(thresholds, None, 0.9999999))

            kendall_min = []
            kendall_max = []
            scaling_min = []
            scaling_max = []
            ratio_min = []
            ratio_max = []

            for t in thresholds:
                g = grouped.get_group(t)
                kendall_min.append(g["kendall_corr"].min())
                kendall_max.append(g["kendall_corr"].max())
                scaling_min.append(g["scaling_factor"].clip(0, 40).min())
                scaling_max.append(g["scaling_factor"].clip(0, 40).max())
                ratio_min.append(g["ratio_mean"].min())
                ratio_max.append(g["ratio_mean"].max())

            # Kendall
            ax1.fill_between(transformed_x, kendall_min, kendall_max, color=colors["kendall"], alpha=0.3)
            ax1.plot(transformed_x, [(lo + hi) / 2 for lo, hi in zip(kendall_min, kendall_max)], color=colors["kendall"], linewidth=2)
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel("Kendall", color=colors["kendall"])
            ax1.tick_params(axis='y', labelcolor=colors["kendall"], labelsize=6)

            # Scaling
            ax2.fill_between(transformed_x, scaling_min, scaling_max, color=colors["scaling"], alpha=0.3)
            ax2.plot(transformed_x, [(lo + hi) / 2 for lo, hi in zip(scaling_min, scaling_max)], color=colors["scaling"], linewidth=2, linestyle=":")
            ax2.set_ylim(0, 40)
            ax2.set_ylabel("Scaling", color=colors["scaling"])
            ax2.tick_params(axis='y', labelcolor=colors["scaling"], labelsize=6)

            # Ratio
            ax3.fill_between(transformed_x, ratio_min, ratio_max, color=colors["ratio"], alpha=0.3)
            ax3.plot(transformed_x, [(lo + hi) / 2 for lo, hi in zip(ratio_min, ratio_max)], color=colors["ratio"], linewidth=2, linestyle="--")
            ax3.set_yscale('log')
            ax3.set_ylim(1e-2, 10)
            ax3.set_ylabel("Out/In Season Ratio", color=colors["ratio"])
            ax3.tick_params(axis='y', labelcolor=colors["ratio"], labelsize=6)

            ax1.set_title(f"{model}", fontsize=10)
            ax1.set_xlabel("-log(1 - threshold)", fontsize=8)
            ax1.set_xticks(transformed_x)
            thresholds_to_hide_labels = {"0.2", "0.75", "0.85", "0.91", "0.92", "0.93", "0.94", "0.95", "0.96", "0.97"}
            ax1.set_xticklabels(
                [str(t) if str(t) not in thresholds_to_hide_labels else "" for t in thresholds],
                rotation=90,
                fontsize=6
            )
            ax1.grid(True, linestyle=":", linewidth=0.5)

        # Legend for the selected-models envelopes
        handles = [
            Line2D([], [], color=colors["kendall"], linestyle="-", label="Kendall"),
            Line2D([], [], color=colors["scaling"], linestyle=":", label="Scaling Factor"),
            Line2D([], [], color=colors["ratio"], linestyle="--", label="Out/In Season Ratio"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        pdf.savefig(fig)
        print(f"Saved envelope plot for species: {species}")
        plt.close(fig)

print(f"Final filtered PDF saved to: {output_minmax_dir}/species_minmax_metrics_selected_models_3.pdf")
