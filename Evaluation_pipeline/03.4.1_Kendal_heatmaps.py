"""
03.4.1_Kendal_heatmaps.py
=========================

Purpose
-------
From per-day reference and prediction counts stored in JSON (one JSON per model),
compute Kendall correlations between predicted taxa (by threshold) and reference
taxa, then produce:

1) Long-form CSV tables per validation dataset and model:
   03_correlation/03.4_All_corr_tables/<DATASET>_<MODEL>_species_correlation_longform.csv

2) Heatmaps (one panel for reference taxa, one panel for Raindrops at threshold 0)
   for selected species/thresholds you define in `models_with_thresholds_and_metadata`.
   Saved as PDF and PNG in:
   03_correlation/03.4_All_corr_tables/04.2_plots/

3) An optional “average over datasets” heatmap per model (mean correlation
   over all processed datasets), using the same species/thresholds list.

Inputs
------
- A list of JSON files produced earlier in the pipeline (see 01_build_dict_inference.py).
  Each JSON is expected to have the shape:
    {
      "<model_name>": {
        "<validation_dataset>": {
          "<species>": {
            "<threshold_as_string>": {
               "reference_counts": {"YYYY-MM-DD": int | float | None, ...},
               "prediction_counts": {"YYYY-MM-DD": int | float | None, ...}
            },
            ...
          },
          ...
        },
        ...
      }
    }

- The dictionary `models_with_thresholds_and_metadata` describing, for each model
  and dataset, which species to plot and at which confidence threshold.

Outputs
-------
- Long-form CSVs with columns:
    pred_species, ref_species, threshold, correlation, ref_ref_corr, high_ref_ref_corr
  where:
    * correlation: Kendall tau between the prediction series (at a given threshold)
      and the reference series.
    * ref_ref_corr: Kendall tau between the reference series of the two taxa;
      used to flag pairs with strong reference-level coherence.
    * high_ref_ref_corr: True if ref_ref_corr > 0.5 (put a star in the annotation).

- Heatmap figures (PDF + PNG) per dataset/model and also averaged over datasets.

Dependencies
------------
- pandas
- numpy
- scipy
- seaborn
- matplotlib

Notes and assumptions
---------------------
- Correlations are computed on the intersection of dates where both series
  are available; if fewer than 2 common points, correlation is NaN.
- For thresholds: we build the prediction dataframe from threshold '0' (as provided)
  and then zero-out values below each tested threshold. This mirrors daily
  “kept vs discarded” behavior at the count level.
- When selecting rows for a given threshold in the plotting function, we use
  `np.isclose` rather than `==` to avoid float-equality problems.
- If "Raindrops" predictions at threshold 0 are present, a second heatmap panel
  shows correlation against “Raindrops_0”.
"""

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# ------------------------- Configuration -------------------------

# Where to store CSV tables and plots
output_table_dir = "03_correlation/03.4_All_corr_tables"
os.makedirs(output_table_dir, exist_ok=True)
output_plot_dir = os.path.join(output_table_dir, "04.2_plots")
os.makedirs(output_plot_dir, exist_ok=True)

# JSON files to process (add/remove as needed)
json_paths = [
    '01_json_files/dict_2024_PBS_2025Q1_Beta_onnx_mixed_15sp.json',
    '01_json_files/dict_2024_PBU_2025Q1_Beta_onnx_mixed_15sp.json',
    '01_json_files/dict_2024_PPY_2025Q1_Beta_onnx_mixed_15sp.json',
    '01_json_files/dict_2024_PNE_2025Q1_Beta_onnx_mixed_15sp.json',
    '01_json_files/dict_2024_PLZ_2025Q1_Beta_onnx_mixed_15sp.json',
    '01_json_files/dict_2024_PBS_2025Q2_Gamma_onnx_mixed_15sp_redFluo.json',
    '01_json_files/dict_2024_PBU_2025Q2_Gamma_onnx_mixed_15sp_redFluo.json',
    '01_json_files/dict_2024_PPY_2025Q2_Gamma_onnx_mixed_15sp_redFluo.json',
    '01_json_files/dict_2024_PNE_2025Q2_Gamma_onnx_mixed_15sp_redFluo.json',
    '01_json_files/dict_2024_PLZ_2025Q2_Gamma_onnx_mixed_15sp_redFluo.json',
]

# Thresholds to compute correlations for
thresholds_to_test = [
    0, 0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94,
    0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999
]

# Species and thresholds to display in the heatmaps (per model and dataset)
# These do not change how correlations are computed; they only control which
# rows and thresholds are shown in the figure.
# Following values are exemples
models_with_thresholds_and_metadata = {
    "2025Q1_Beta_onnx_mixed_15sp": {
        "2024_PBU": {"Alnus": 0.9, "Betula": 0.9, "Corylus": 0.9, "Poaceae": 0.92, "Raindrops": 0},
        "2024_PBS": {"Alnus": 0.9, "Betula": 0.9, "Corylus": 0.9, "Poaceae": 0.92, "Raindrops": 0},
        "2024_PNE": {"Alnus": 0.9, "Betula": 0.9, "Corylus": 0.9, "Poaceae": 0.92, "Raindrops": 0},
        "2024_PPY": {"Alnus": 0.9, "Betula": 0.9, "Corylus": 0.9, "Poaceae": 0.92, "Raindrops": 0},
        "2024_PLZ": {"Alnus": 0.9, "Betula": 0.9, "Corylus": 0.9, "Poaceae": 0.92, "Raindrops": 0},
    },
    "2025Q2_Gamma_onnx_mixed_15sp_redFluo": {
        "2024_PBU": {"Alnus": 0.85, "Betula": 0.95, "Corylus": 0.97, "Poaceae": 0.97, "Raindrops": 0},
        "2024_PBS": {"Alnus": 0.85, "Betula": 0.95, "Corylus": 0.97, "Poaceae": 0.97, "Raindrops": 0},
        "2024_PNE": {"Alnus": 0.85, "Betula": 0.95, "Corylus": 0.97, "Poaceae": 0.97, "Raindrops": 0},
        "2024_PPY": {"Alnus": 0.85, "Betula": 0.95, "Corylus": 0.97, "Poaceae": 0.97, "Raindrops": 0},
        "2024_PLZ": {"Alnus": 0.85, "Betula": 0.95, "Corylus": 0.97, "Poaceae": 0.97, "Raindrops": 0},
    },
}

# ------------------------- Helpers -------------------------

def kendall_on_common(a: pd.Series, b: pd.Series) -> float:
    """Compute Kendall tau on the intersection of indices where both series are non-NaN."""
    common = a.dropna().index.intersection(b.dropna().index)
    if len(common) <= 1:
        return np.nan
    val, _ = kendalltau(a.loc[common], b.loc[common])
    return val

def plot_correlation_heatmaps(
    df_corr: pd.DataFrame,
    target_model: str,
    target_dataset: str,
    thresholds_dict: dict,
    output_dir: str,
    atol: float = 1e-12
) -> None:
    """
    Build two heatmaps:
      - Left: predicted species (rows) vs reference species (cols), at the per-species threshold.
      - Right: correlation of each predicted species vs Raindrops at threshold 0 (single column).

    Uses np.isclose to select thresholded rows to avoid float-equality traps.
    """
    # Species list and labels on the Y axis
    target_species_list = list(thresholds_dict.keys())

    # Build correlation matrices for reference species and raindrops
    ref_species_full = sorted(set(df_corr["ref_species"]) - {"Raindrops_0"})
    ref_corr_matrix = pd.DataFrame(index=target_species_list, columns=ref_species_full, dtype=float)
    annot_matrix = pd.DataFrame(index=target_species_list, columns=ref_species_full, dtype=object)
    raindrop_corr_series = pd.Series(index=target_species_list, dtype=float)

    # Fill matrices row by row (species by species)
    for pred_species in target_species_list:
        desired_th = thresholds_dict.get(pred_species, 0)
        # Select rows for this species at the requested threshold with tolerance
        rows_species = df_corr[df_corr["pred_species"] == pred_species]
        mask_th = np.isclose(rows_species["threshold"].values.astype(float), float(desired_th), atol=atol)
        rows_species = rows_species.loc[rows_species.index[mask_th]]

        # Reference species block
        ref_rows = rows_species[rows_species["ref_species"] != "Raindrops_0"]
        for _, r in ref_rows.iterrows():
            value = r["correlation"]
            high_ref = bool(r["high_ref_ref_corr"])
            ref_corr_matrix.loc[pred_species, r["ref_species"]] = value
            annot_matrix.loc[pred_species, r["ref_species"]] = (
                f"{value:.2f}{'*' if high_ref and not pd.isna(value) else ''}"
                if not pd.isna(value) else ""
            )

        # Raindrops column (if any)
        rain_rows = rows_species[rows_species["ref_species"] == "Raindrops_0"]
        if not rain_rows.empty:
            raindrop_corr_series.loc[pred_species] = rain_rows.iloc[0]["correlation"]

    # Reorder columns so that overlapping species (pred also in ref) appear first
    overlap = [s for s in ref_corr_matrix.columns if s in target_species_list]
    others = [s for s in ref_corr_matrix.columns if s not in overlap]
    ref_corr_matrix = ref_corr_matrix[overlap + others]
    annot_matrix = annot_matrix[overlap + others]

    # Label rows with species and threshold, e.g. "Betula (0.9)"
    row_labels = [f"{s} ({thresholds_dict.get(s, 0)})" for s in ref_corr_matrix.index]
    ref_corr_matrix.index = row_labels
    annot_matrix.index = row_labels
    raindrop_corr_series.index = row_labels

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [5, 1]})

    sns.heatmap(
        ref_corr_matrix.astype(float),
        annot=annot_matrix, fmt="",
        cmap="magma", center=0, vmin=-1, vmax=1,
        ax=axes[0], xticklabels=True, yticklabels=True,
        linewidths=0.5, square=True
    )
    axes[0].set_title(f"Correlation with Reference Species\nModel: {target_model} | Dataset: {target_dataset}")
    axes[0].set_xlabel("Reference Species")
    axes[0].set_ylabel("Predicted Species (threshold)")

    if not raindrop_corr_series.dropna().empty:
        sns.heatmap(
            raindrop_corr_series.to_frame(name="Raindrops_0").astype(float),
            annot=True, fmt=".2f", cmap="magma", center=0, vmin=-1, vmax=1,
            ax=axes[1], xticklabels=True, yticklabels=True,
            cbar=True, linewidths=0.5, square=True
        )
        axes[1].set_title(f"Correlation with Raindrops (Threshold 0)\nModel: {target_model}")
        axes[1].set_xlabel("Raindrops (0)")
        axes[1].set_ylabel("")
    else:
        axes[1].axis("off")

    plt.tight_layout()
    filename_base = f"{target_dataset}_{target_model}_selected_species_correlation_heatmap"
    plt.savefig(os.path.join(output_dir, filename_base + ".pdf"))
    plt.savefig(os.path.join(output_dir, filename_base + ".png"), dpi=300)
    plt.close(fig)

# ------------------------- Main computation -------------------------

all_results_by_model = {}  # to build “average over datasets” per model

for json_path in json_paths:
    with open(json_path, "r") as f:
        model_blob = json.load(f)

    # Expect a single top-level model name in each file
    model_name = list(model_blob.keys())[0]
    datasets = model_blob[model_name].keys()

    for dataset_name in datasets:
        # Build wide tables for predictions (at threshold "0") and references
        pred_raw = {}
        ref_raw = {}

        for species, species_thresholds in model_blob[model_name][dataset_name].items():
            # Require threshold "0" to get the base daily series
            if "0" not in species_thresholds:
                continue

            # Prediction series at threshold 0
            for d, v in species_thresholds["0"].get("prediction_counts", {}).items():
                pred_raw.setdefault(species, {})[d] = v

            # Reference series
            for d, v in species_thresholds["0"].get("reference_counts", {}).items():
                ref_raw.setdefault(species, {})[d] = v

        df_pred_raw = pd.DataFrame(pred_raw).sort_index()
        df_ref = pd.DataFrame(ref_raw).sort_index()

        # Keep only species present in both tables
        common_species = [c for c in df_ref.columns if c in df_pred_raw.columns]
        df_ref = df_ref[common_species]
        df_pred_raw = df_pred_raw[common_species]

        # Raindrops 0-series, if present in predictions
        raindrops_0 = df_pred_raw["Raindrops"] if "Raindrops" in df_pred_raw.columns else None

        # Reference-to-reference Kendall dictionary (for annotation stars)
        ref_ref_corr = {}
        for s1 in df_ref.columns:
            for s2 in df_ref.columns:
                ref_ref_corr[(s1, s2)] = kendall_on_common(df_ref[s1], df_ref[s2])

        # Build the long-form correlation table across thresholds and species pairs
        rows = []
        for th in thresholds_to_test:
            # Apply threshold to predictions: keep value if >= th else set to 0
            df_pred_th = df_pred_raw.applymap(lambda x: x if (pd.notna(x) and x >= th) else 0)

            # Correlation against each reference species
            for pred_sp in df_pred_th.columns:
                for ref_sp in df_ref.columns:
                    corr = kendall_on_common(df_pred_th[pred_sp], df_ref[ref_sp])
                    ref_to_ref = ref_ref_corr.get((pred_sp, ref_sp), np.nan)
                    rows.append({
                        "pred_species": pred_sp,
                        "ref_species": ref_sp,
                        "threshold": float(th),
                        "correlation": corr,
                        "ref_ref_corr": ref_to_ref,
                        "high_ref_ref_corr": (ref_to_ref > 0.5) if not np.isnan(ref_to_ref) else False,
                    })

                # Correlation against raindrops (threshold 0) if available
                if raindrops_0 is not None:
                    corr_r = kendall_on_common(df_pred_th[pred_sp], raindrops_0)
                    rows.append({
                        "pred_species": pred_sp,
                        "ref_species": "Raindrops_0",
                        "threshold": float(th),
                        "correlation": corr_r,
                        "ref_ref_corr": np.nan,
                        "high_ref_ref_corr": False,
                    })

        df_long = pd.DataFrame(rows)
        all_results_by_model.setdefault(model_name, []).append(df_long)

        # Save per-dataset long table
        out_csv = os.path.join(output_table_dir, f"{dataset_name}_{model_name}_species_correlation_longform.csv")
        df_long.to_csv(out_csv, index=False)

        # Plot per-dataset heatmap if thresholds are provided for this model/dataset
        th_dict = models_with_thresholds_and_metadata.get(model_name, {}).get(dataset_name, {})
        if th_dict:
            plot_correlation_heatmaps(df_long, model_name, dataset_name, th_dict, output_plot_dir)

# Average heatmaps across datasets for each model (mean correlations per (pred, ref, threshold))
for model_name, list_of_df in all_results_by_model.items():
    df_all = pd.concat(list_of_df, ignore_index=True)

    # Choose a species/threshold dict to label rows (first available dataset entry if present)
    # This only controls which species/thresholds appear in the figure, not the averaging itself.
    maybe_any = models_with_thresholds_and_metadata.get(model_name, {})
    thresholds_dict = next(iter(maybe_any.values()), {}) if maybe_any else {}

    df_avg = (
        df_all
        .groupby(["pred_species", "ref_species", "threshold"], as_index=False)
        .agg({
            "correlation": "mean",
            "ref_ref_corr": "mean",
            "high_ref_ref_corr": "max"
        })
    )

    if thresholds_dict:
        plot_correlation_heatmaps(df_avg, model_name, "AVG_OVER_DATASETS", thresholds_dict, output_plot_dir)
