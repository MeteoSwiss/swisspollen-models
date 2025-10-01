"""
05.1_plotting_SF_timeseries.py
===============================

Purpose
-------
1) Extract the scaling factors (SF) for specific (species, model, dataset, threshold)
   combinations from the table produced in step 03.1, and save a filtered CSV.
2) Plot time series (reference vs prediction and scaled prediction) per combination,
   using the chosen threshold and per-dataset scaling factor.
3) Compute averaged scaling factors by species and model (excluding very large SFs),
   then plot time series again using the averaged SF.
4) Produce a multi-page PDF that arranges plots by species with all (model × dataset)
   panels on one page, and a separate PDF showing the (scaled prediction − reference)
   difference curves with consistent y-limits across pages.

Inputs
------
- CSV from step 03.1:
    03_correlation/03.1_tables/table_thresholds_corrK_SF_SigToNoise.csv
  Expected columns include at least:
    species, model, dataset, threshold, scaling_factor, season_start, season_end

- JSON files with daily counts built earlier in the pipeline (01_build_dict_inference.py).
  Each file has structure:
    {
      "<model>": {
        "<validation_dataset>": {
          "<species>": {
            "<threshold_as_string>": {
              "reference_counts": {"YYYY-MM-DD": number or None, ...},
              "prediction_counts": {"YYYY-MM-DD": number or None, ...}
            },
            ...
          }
        }
      }
    }

- A dictionary `thresholds_by_model_dataset_species` that declares, for each model and dataset,
  which species to plot and which confidence threshold to use.

Outputs
-------
- 05_scaling/scaling_factors_filtered.csv
    Filtered table with one row per requested (species, model, dataset, threshold)
    combination and its scaling_factor.

- 05_scaling/averaged_scaling_factors.csv
    Mean scaling factors per (species, model), restricted to SF ≤ 40.

- HTML time series plots:
    05_scaling/05.5_visual_timeseries/05.1_notAveraged/*.html
    05_scaling/05.5_visual_timeseries/05.1_Averaged/*.html

- Multi-page PDFs:
    05_scaling/summary_averaged_scaling_plots.pdf
      Grid by species: rows=models, cols=datasets; reference vs prediction (scaled with averaged SF).
    05_scaling/summary_diff_scaled_vs_ref.pdf
      One page per species showing (scaled prediction − reference) curves for all datasets.

Parameters to edit
------------------
- `json_paths`: list the JSONs you want to include.
- `thresholds_by_model_dataset_species`: choose species and thresholds per dataset.
- Fixed date range for summary plots (see `date_range` if you want a different window).

Dependencies
------------
- pandas
- numpy
- matplotlib
- plotly
- (optional) scipy is not required in this file

Notes
-----
- Threshold matching in the JSON uses string keys (e.g., "0.9"); in plots we convert the
  chosen float threshold to a string for indexing.
- Missing dates are handled by reindexing; None values in Plotly render as gaps.
- Raindrops can be assigned an SF of 1 so it always plots on the same scale.
"""

import os
import json
import pandas as pd
import numpy as np

# ----------------------------- Paths and inputs -----------------------------

# Source table from step 03.1
csv_path = "03_correlation/03.1_tables/table_thresholds_corrK_SF_SigToNoise.csv"
df = pd.read_csv(csv_path)

# JSON files to load (add/remove as needed)
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

# Where to write plots
output_dir_notAveraged = "05_scaling/05.5_visual_timeseries/05.1_notAveraged"
output_dir_Averaged = "05_scaling/05.5_visual_timeseries/05.1_Averaged"
os.makedirs(output_dir_notAveraged, exist_ok=True)
os.makedirs(output_dir_Averaged, exist_ok=True)

# Which species and thresholds to visualize (edit as needed)
thresholds_by_model_dataset_species = {
    "2025Q1_Beta_onnx_mixed_15sp": {
        "2024_PBU": {"Alnus": 0.9, "Betula": 0.9, "Corylus": 0.9, "Poaceae": 0.92, "Fraxinus": 0.999, "Quercus": 0.6, "Fagus": 0.999, "Raindrops": 0},
        "2024_PBS": {"Alnus": 0.9, "Betula": 0.9, "Corylus": 0.9, "Poaceae": 0.92, "Fraxinus": 0.999, "Quercus": 0.6, "Fagus": 0.999, "Raindrops": 0},
        "2024_PNE": {"Alnus": 0.9, "Betula": 0.9, "Corylus": 0.9, "Poaceae": 0.92, "Fraxinus": 0.999, "Quercus": 0.6, "Fagus": 0.999, "Raindrops": 0},
        "2024_PPY": {"Alnus": 0.9, "Betula": 0.9, "Corylus": 0.9, "Poaceae": 0.92, "Fraxinus": 0.999, "Quercus": 0.6, "Fagus": 0.999, "Raindrops": 0},
        "2024_PLZ": {"Alnus": 0.9, "Betula": 0.9, "Corylus": 0.9, "Poaceae": 0.92, "Fraxinus": 0.999, "Quercus": 0.6, "Fagus": 0.999, "Raindrops": 0}
    }
}

# ----------------------------- Build filtered SF table -----------------------------

# Flatten the thresholds dictionary to a query table
query_rows = []
for model, dataset_dict in thresholds_by_model_dataset_species.items():
    for dataset, species_dict in dataset_dict.items():
        for species, threshold in species_dict.items():
            query_rows.append({"species": species, "model": model, "dataset": dataset, "threshold": threshold})

query_df = pd.DataFrame(query_rows)

# Join with the full metrics table to pull the scaling_factor
merged_df = pd.merge(
    query_df,
    df,
    on=["species", "model", "dataset", "threshold"],
    how="left"
)

# Keep the SF and identifying columns
result_df = merged_df[["species", "model", "dataset", "threshold", "scaling_factor"]]

# Save the filtered SFs
sf_filtered_out = "05_scaling/scaling_factors_filtered.csv"
os.makedirs(os.path.dirname(sf_filtered_out), exist_ok=True)
result_df.to_csv(sf_filtered_out, index=False)
print(f"Filtered scaling factors written to: {sf_filtered_out}")

# ----------------------------- Load JSON and plot (per-dataset SF) -----------------------------

import plotly.graph_objects as go  # plotly is used for interactive HTML timeseries

def load_json_data(jsons):
    """Merge multiple model JSON files into a single nested dict [model][dataset][species][threshold]."""
    combined = {}
    for path in jsons:
        with open(path, "r") as f:
            data = json.load(f)
        for model_name, model_data in data.items():
            combined.setdefault(model_name, {})
            for ds_name, ds_data in model_data.items():
                combined[model_name].setdefault(ds_name, {})
                for species, species_dict in ds_data.items():
                    combined[model_name][ds_name].setdefault(species, {})
                    for thr, thr_dict in species_dict.items():
                        combined[model_name][ds_name][species].setdefault(thr, {})
                        for key in ["reference_counts", "prediction_counts"]:
                            existing = combined[model_name][ds_name][species][thr].get(key, {})
                            newvals = thr_dict.get(key, {})
                            for dt, val in newvals.items():
                                if dt not in existing:
                                    existing[dt] = val
                            combined[model_name][ds_name][species][thr][key] = existing
    return combined

def plot_prediction_and_reference(data, scaling_df, thresholds_dict, output_dir):
    """Plot reference, raw prediction, and prediction × per-dataset SF, saving HTML files."""
    os.makedirs(output_dir, exist_ok=True)
    scaling_df = scaling_df.copy()
    scaling_df["threshold_str"] = scaling_df["threshold"].astype(str)

    for model, dataset_dict in thresholds_dict.items():
        for dataset, species_dict in dataset_dict.items():
            for species, threshold in species_dict.items():
                t_str = str(threshold)
                try:
                    pred_counts = data[model][dataset][species][t_str]["prediction_counts"]
                    ref_counts = data[model][dataset][species]["0"]["reference_counts"]
                except KeyError as e:
                    print(f"Missing JSON for {species} / {model} / {dataset} / threshold {t_str}: {e}")
                    continue

                match = scaling_df[
                    (scaling_df["species"] == species) &
                    (scaling_df["model"] == model) &
                    (scaling_df["dataset"] == dataset) &
                    (scaling_df["threshold_str"] == t_str)
                ]
                if match.empty:
                    print(f"No scaling factor in CSV for {species} / {model} / {dataset} / {t_str}")
                    continue

                sf = match["scaling_factor"].iloc[0]

                common_dates = sorted(set(pred_counts) & set(ref_counts))
                raw_preds = [pred_counts[d] for d in common_dates]
                scaled_preds = [p * sf if p is not None else None for p in raw_preds]
                refs = [ref_counts[d] for d in common_dates]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=common_dates, y=refs, name="Reference (T=0)", line=dict(color="black")))
                fig.add_trace(go.Scatter(x=common_dates, y=raw_preds, name="Prediction", line=dict(dash="dot")))
                fig.add_trace(go.Scatter(x=common_dates, y=scaled_preds, name=f"Scaled ×{sf:.2f}"))

                fig.update_layout(
                    title=f"{species} – {model} – {dataset}",
                    xaxis_title="Date",
                    yaxis_title="Count",
                    template="plotly_white"
                )

                fname = f"{species}_{model}_{dataset}_timeseries.html".replace("/", "_")
                fig.write_html(os.path.join(output_dir, fname))
                print(f"Saved: {os.path.join(output_dir, fname)}")

# Load JSON data and the SF table we just saved
data = load_json_data(json_paths)
scaling_df = pd.read_csv(csv_path)  # full table (has all SF rows)
plot_prediction_and_reference(data, scaling_df, thresholds_by_model_dataset_species, output_dir_notAveraged)

# ----------------------------- Averaged SF computation -----------------------------

def compute_average_scaling_factors(scaling_df, thresholds_dict, output_path, max_sf=40):
    """
    Average scaling factors by (species, model) across datasets, keeping only SF ≤ max_sf.
    Writes CSV and returns the averaged DataFrame.
    """
    scaling_df = scaling_df.copy()
    scaling_df["threshold_str"] = scaling_df["threshold"].astype(str)

    rows = []
    for model, dataset_dict in thresholds_dict.items():
        for dataset, species_dict in dataset_dict.items():
            for species, threshold in species_dict.items():
                t_str = str(threshold)
                match = scaling_df[
                    (scaling_df["species"] == species) &
                    (scaling_df["model"] == model) &
                    (scaling_df["threshold_str"] == t_str)
                ]
                if match.empty:
                    print(f"No SF rows for {species} / {model} / threshold {t_str}")
                    continue

                for _, r in match.iterrows():
                    if pd.notna(r["scaling_factor"]) and r["scaling_factor"] <= max_sf:
                        rows.append({"species": r["species"], "model": r["model"], "scaling_factor": r["scaling_factor"]})

    avg_df = pd.DataFrame(rows).groupby(["species", "model"], as_index=False).mean(numeric_only=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    avg_df.to_csv(output_path, index=False)
    print(f"Averaged scaling factors written to: {output_path}")
    return avg_df

avg_scaling_path = "05_scaling/averaged_scaling_factors.csv"
avg_scaling_df = compute_average_scaling_factors(scaling_df, thresholds_by_model_dataset_species, avg_scaling_path)

# ----------------------------- Plot with averaged SF -----------------------------

def plot_with_average_scaling(data, avg_sf_df, thresholds_dict, output_dir):
    """Plot reference and prediction × averaged SF for each requested combination."""
    os.makedirs(output_dir, exist_ok=True)

    for model, dataset_dict in thresholds_dict.items():
        for dataset, species_dict in dataset_dict.items():
            for species, threshold in species_dict.items():
                t_str = str(threshold)
                try:
                    pred_counts = data[model][dataset][species][t_str]["prediction_counts"]
                    ref_counts = data[model][dataset][species]["0"]["reference_counts"]
                except KeyError as e:
                    print(f"Missing JSON for {species} / {model} / {dataset} / threshold {t_str}: {e}")
                    continue

                match = avg_sf_df[(avg_sf_df["species"] == species) & (avg_sf_df["model"] == model)]
                if match.empty:
                    print(f"No averaged SF for {species} / {model}")
                    continue

                sf = match["scaling_factor"].iloc[0]

                common_dates = sorted(set(pred_counts) & set(ref_counts))
                raw_preds = [pred_counts[d] for d in common_dates]
                scaled_preds = [p * sf if p is not None else None for p in raw_preds]
                refs = [ref_counts[d] for d in common_dates]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=common_dates, y=refs, name="Reference (T=0)", line=dict(color="black")))
                fig.add_trace(go.Scatter(x=common_dates, y=scaled_preds, name=f"Avg Scaled ×{sf:.2f}"))
                fig.add_trace(go.Scatter(x=common_dates, y=raw_preds, name="Prediction (Raw)", line=dict(dash="dot")))
                fig.update_layout(
                    title=f"{species} – {model} – {dataset} (Avg SF)",
                    xaxis_title="Date",
                    yaxis_title="Count",
                    template="plotly_white"
                )

                fname = f"{species}_{model}_{dataset}_avg_scaled.html".replace("/", "_")
                fig.write_html(os.path.join(output_dir, fname))
                print(f"Saved avg-scaled plot: {os.path.join(output_dir, fname)}")

plot_with_average_scaling(data, avg_scaling_df, thresholds_by_model_dataset_species, output_dir_Averaged)

# ----------------------------- Persist thresholds and merged JSON -----------------------------

thresholds_path = "05_scaling/thresholds_dict.txt"
json_out_path = "05_scaling/preprocessed_combined_data.json"
os.makedirs(os.path.dirname(thresholds_path), exist_ok=True)

with open(thresholds_path, "w") as f:
    f.write(repr(thresholds_by_model_dataset_species))

with open(json_out_path, "w") as f:
    json.dump(data, f)

print(f"Saved thresholds to: {thresholds_path}")
print(f"Saved combined JSON to: {json_out_path}")

# ----------------------------- Multi-page summary PDFs -----------------------------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

# Load back the saved artifacts for clarity of workflow
with open(thresholds_path, "r") as f:
    thresholds_by_model_dataset_species = eval(f.read())
with open(json_out_path, "r") as f:
    combined_data = json.load(f)
avg_scaling_df = pd.read_csv(avg_scaling_path)

# Ensure Raindrops has an SF of 1 per model
if "Raindrops" in avg_scaling_df["species"].unique():
    avg_scaling_df.loc[avg_scaling_df["species"] == "Raindrops", "scaling_factor"] = 1
else:
    for model in avg_scaling_df["model"].unique():
        avg_scaling_df = pd.concat(
            [avg_scaling_df, pd.DataFrame([{"species": "Raindrops", "model": model, "scaling_factor": 1}])],
            ignore_index=True
        )

species_list = sorted({sp for model in thresholds_by_model_dataset_species.values() for ds in model.values() for sp in ds})
models = sorted(thresholds_by_model_dataset_species.keys())
datasets = sorted({ds for model in thresholds_by_model_dataset_species.values() for ds in model.keys()})

# 1) Summary grid with averaged SF
pdf_output = "05_scaling/summary_averaged_scaling_plots.pdf"
os.makedirs(os.path.dirname(pdf_output), exist_ok=True)

with PdfPages(pdf_output) as pp:
    for species in species_list:
        fig, axes = plt.subplots(nrows=len(models), ncols=len(datasets), figsize=(4 * len(datasets), 3 * len(models)), squeeze=False)
        fig.suptitle(f"Species: {species}", fontsize=16)

        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                ax = axes[i][j]

                threshold_dict = thresholds_by_model_dataset_species.get(model, {}).get(dataset, {})
                if species not in threshold_dict:
                    ax.set_title(f"{model} / {dataset} (no data)", fontsize=8)
                    ax.axis("off")
                    continue

                t_str = str(threshold_dict[species])
                try:
                    pred_counts = combined_data[model][dataset][species][t_str]["prediction_counts"]
                    ref_counts = combined_data[model][dataset][species]["0"]["reference_counts"]
                except KeyError:
                    ax.set_title(f"{model} / {dataset} (missing JSON)", fontsize=8)
                    ax.axis("off")
                    continue

                match = avg_scaling_df[(avg_scaling_df["species"] == species) & (avg_scaling_df["model"] == model)]
                if match.empty:
                    ax.set_title(f"{model} / {dataset} (no SF)", fontsize=8)
                    ax.axis("off")
                    continue

                sf = match["scaling_factor"].iloc[0]

                common_dates = sorted(set(pred_counts) & set(ref_counts))
                if not common_dates:
                    ax.set_title(f"{model} / {dataset} (no overlap)", fontsize=8)
                    ax.axis("off")
                    continue

                dates = pd.to_datetime(common_dates)
                raw_preds = [pred_counts[d] for d in common_dates]
                scaled_preds = [p * sf if p is not None else None for p in raw_preds]
                refs = [ref_counts[d] for d in common_dates]

                ax.plot(dates, refs, color="black", label="Ref (T=0)", linewidth=0.8)
                ax.plot(dates, raw_preds, linestyle=":", color="#FC6C85", label="Raw Pred", linewidth=0.8)
                ax.plot(dates, scaled_preds, color="#E69F00", label=f"Scaled ×{sf:.2f}", linewidth=0.8)

                ax.set_title(f"{model} – {dataset}", fontsize=8)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.tick_params(axis='x', rotation=30, labelsize=6)
                ax.tick_params(axis='y', labelsize=6)
                ax.set_ylabel("Count", fontsize=7)
                # Set a common window if desired; adjust as needed
                ax.set_xlim(pd.to_datetime("2024-01-01"), pd.to_datetime("2024-09-15"))

                # Add a legend in the last column of each row if present
                handles, labels = ax.get_legend_handles_labels()
                if handles and j == len(datasets) - 1:
                    ax.legend(handles, labels, loc="upper left", fontsize=6)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        pp.savefig(fig)
        plt.close(fig)

print(f"PDF saved: {pdf_output}")

# 2) Difference curves (scaled prediction − reference) with global Y limits
pdf_diff_output = "05_scaling/summary_diff_scaled_vs_ref.pdf"
date_range = pd.date_range("2024-01-01", "2024-09-15", freq="D")

# Scan global min/max across all panels for consistent y-limits
global_min = float("inf")
global_max = float("-inf")

for species in species_list:
    for model in models:
        for dataset in thresholds_by_model_dataset_species.get(model, {}):
            if species not in thresholds_by_model_dataset_species[model][dataset]:
                continue
            t_str = str(thresholds_by_model_dataset_species[model][dataset][species])
            try:
                pred_counts = data[model][dataset][species][t_str]["prediction_counts"]
                ref_counts = data[model][dataset][species]["0"]["reference_counts"]
            except KeyError:
                continue

            match = avg_scaling_df[(avg_scaling_df["species"] == species) & (avg_scaling_df["model"] == model)]
            if match.empty:
                continue
            sf = match["scaling_factor"].iloc[0]

            pred = pd.Series(pred_counts)
            pred.index = pd.to_datetime(pred.index)
            pred = pred.reindex(date_range).astype(float) * sf

            ref = pd.Series(ref_counts)
            ref.index = pd.to_datetime(ref.index)
            ref = ref.reindex(date_range).astype(float)

            diff = (pred - ref).dropna()
            if not diff.empty:
                global_min = min(global_min, diff.min())
                global_max = max(global_max, diff.max())

# Add a small padding
if np.isfinite(global_min) and np.isfinite(global_max):
    padding = 0.05 * (global_max - global_min) if global_max > global_min else 1.0
    global_min -= padding
    global_max += padding
else:
    global_min, global_max = -1, 1  # fallback

with PdfPages(pdf_diff_output) as pdf:
    for species in species_list:
        fig, ax = plt.subplots(figsize=(8, 3))
        # Assign a color per dataset (cycled)
        import matplotlib.cm as cm
        color_map = cm.get_cmap("tab10")
        dataset_colors = {}

        for model in models:
            for i, dataset in enumerate(thresholds_by_model_dataset_species.get(model, {})):
                if species not in thresholds_by_model_dataset_species[model][dataset]:
                    continue
                t_str = str(thresholds_by_model_dataset_species[model][dataset][species])
                try:
                    pred_counts = data[model][dataset][species][t_str]["prediction_counts"]
                    ref_counts = data[model][dataset][species]["0"]["reference_counts"]
                except KeyError:
                    continue

                match = avg_scaling_df[(avg_scaling_df["species"] == species) & (avg_scaling_df["model"] == model)]
                if match.empty:
                    continue
                sf = match["scaling_factor"].iloc[0]

                if dataset not in dataset_colors:
                    dataset_colors[dataset] = color_map(i % 10)
                color = dataset_colors[dataset]

                pred = pd.Series(pred_counts)
                pred.index = pd.to_datetime(pred.index)
                pred = pred.reindex(date_range).astype(float) * sf

                ref = pd.Series(ref_counts)
                ref.index = pd.to_datetime(ref.index)
                ref = ref.reindex(date_range).astype(float)

                if pred.dropna().empty or ref.dropna().empty:
                    continue

                diff = pred - ref
                ax.plot(date_range, diff, color=color, linewidth=1.2, alpha=0.9, label=dataset)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{species} – Difference (Pred × SF − Ref)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Difference")
        ax.set_xlim(date_range[0], date_range[-1])
        ax.set_ylim(global_min, global_max)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"PDF saved: {pdf_diff_output}")
