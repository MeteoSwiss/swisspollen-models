"""
03.1.2_correlation_kendall_SF.py
=====================================

Purpose
-------
Build a CSV table that, for each combination of species (taxa), model, validation
dataset, and confidence threshold, reports:
  - an automatically detected season (start/end dates) based on reference counts,
  - an optional manually defined season (if provided),
  - a scaling factor that best aligns predictions to references within the season,
  - simple signal-to-noise indicators (mean inside vs. outside the season),
  - Kendall's tau correlation between reference and prediction daily counts.


Inputs
------
1) A list of JSON files produced by 01_build_dict_inference.py, each following:
   {
     "model_name": {
       "validation_dataset": {
         "species": {
           "<threshold>": {
             "reference_counts": {"YYYY-MM-DD": number | null, ...},
             "prediction_counts": {"YYYY-MM-DD": number | null, ...}
           },
           ...
         },
         ...
       },
       ...
     },
     ...
   }

2) (Optional) A dictionary named `custom_season_dates` that lets you specify
   fixed season start/end dates per (species, validation dataset, model).
   If present for a given triplet, it overrides the automatic season detection.

Outputs
-------
A CSV saved to:
  03_correlation/03.1_tables/table_thresholds_corrK_SF_SigToNoise.csv

The table contains:
  species            Taxa name (JSON key)
  model              Model name
  dataset            Validation dataset
  threshold          Confidence threshold (as float)
  scaling_factor     Best scale to align predictions to references in-season
  season_start       Detected or custom season start date (YYYY-MM-DD)
  season_end         Detected or custom season end date (YYYY-MM-DD)
  mean_in_season     Average predicted daily count within the season
  mean_out_season    Average predicted daily count outside the season
  ratio_mean         mean_out_season / mean_in_season (or inverse if needed)
  kendall_corr       Kendall’s tau on overlapping non-null dates

How season detection works (automatic)
--------------------------------------
find_season_active_dates(reference_counts, threshold=20, min_consecutive_days=7, gap_tolerance=3)

- threshold: minimum reference count for a day to be considered “active”.
- min_consecutive_days: minimum window length used to detect the start of the season.
- gap_tolerance: how many days below the threshold are allowed within a window or
  while extending the season before we stop.

Notes
-----
- Directory creation: the script creates the output folder if needed.
- Dependencies: pandas, numpy, scipy.
- Kendall’s tau returns NaN when there are fewer than 2 valid overlapping dates
  or when one series is constant.
- Reference and prediction dates are used as “YYYY-MM-DD” strings; both series
  are aligned on the intersection of dates for correlation and scaling.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau


def load_json_data(json_paths):
    """
    Read and merge multiple JSON files from 01_build_dict_inference.py into a single
    nested dictionary. If the same (model/dataset/species/threshold) appears in more
    than one file, date -> count pairs are merged without overwriting existing entries.
    """
    combined_data = {}
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for model_name, model_data in data.items():
                if model_name not in combined_data:
                    combined_data[model_name] = {}
                for validation_dataset_name, validation_data in model_data.items():
                    if validation_dataset_name not in combined_data[model_name]:
                        combined_data[model_name][validation_dataset_name] = {}
                    for species, species_data in validation_data.items():
                        if species not in combined_data[model_name][validation_dataset_name]:
                            combined_data[model_name][validation_dataset_name][species] = species_data
                        else:
                            for threshold, threshold_data in species_data.items():
                                if threshold not in combined_data[model_name][validation_dataset_name][species]:
                                    combined_data[model_name][validation_dataset_name][species][threshold] = threshold_data
                                else:
                                    for key in ['reference_counts', 'prediction_counts']:
                                        existing_counts = combined_data[model_name][validation_dataset_name][species][threshold].get(key, {})
                                        new_counts = threshold_data.get(key, {})
                                        for date, count in new_counts.items():
                                            if date not in existing_counts:
                                                existing_counts[date] = count
                                        combined_data[model_name][validation_dataset_name][species][threshold][key] = existing_counts
    return combined_data


def find_season_active_dates(reference_counts, threshold=20, min_consecutive_days=7, gap_tolerance=3):
    """
    Automatically detect a contiguous “active season” from reference daily counts.

    A start index is found by scanning for the first window of length
    `min_consecutive_days` in which the number of sub-threshold or null days does
    not exceed `gap_tolerance`. From that start, the season is extended forward,
    allowing up to `gap_tolerance` sub-threshold/null days before stopping.

    Returns:
        active_dates (set[str]) : Dates (YYYY-MM-DD) inside the detected season
        season_start (datetime) : Start date, or None if not found
        season_end (datetime)   : End date, or None if not found
    """
    dates = sorted(reference_counts.keys())
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    ref_series = [reference_counts[d] for d in dates]

    # Find the first window that passes the active-day rule
    start_index = None
    for i in range(len(ref_series) - min_consecutive_days + 1):
        window = ref_series[i:i + min_consecutive_days]
        valid_points = sum(1 for v in window if v is not None and v >= threshold)
        bad_points = len(window) - valid_points
        if bad_points <= gap_tolerance:
            start_index = i
            break

    if start_index is None:
        return set(), None, None

    # Extend the season forward from start, allowing short gaps
    active_indices = []
    gap_counter = 0
    for j in range(start_index, len(ref_series)):
        val = ref_series[j]
        if val is not None and val >= threshold:
            active_indices.append(j)
            gap_counter = 0
        else:
            gap_counter += 1
            if gap_counter > gap_tolerance:
                break
            active_indices.append(j)

    active_dates = [date_objs[k].strftime("%Y-%m-%d") for k in active_indices]
    return set(active_dates), date_objs[active_indices[0]], date_objs[active_indices[-1]]


def find_best_scaling_factor(reference_counts, prediction_counts, active_dates):
    """
    Compute a scale factor that minimizes the mean squared error between
    reference_counts and scale * prediction_counts, restricted to `active_dates`.

    If no overlapping valid dates exist in-season, returns NaN.
    """
    ref_counts_clean = {d: c for d, c in reference_counts.items() if d in active_dates and c is not None and c > 0}
    pred_counts_clean = {d: c for d, c in prediction_counts.items() if d in active_dates and c is not None}
    common_dates = set(ref_counts_clean) & set(pred_counts_clean)

    if not common_dates:
        return np.nan

    ref_values = np.array([ref_counts_clean[d] for d in common_dates])
    pred_values = np.array([pred_counts_clean[d] for d in common_dates])

    if len(ref_values) == 0 or len(pred_values) == 0:
        return np.nan

    def objective(scale_factor):
        return np.mean((ref_values - scale_factor * pred_values) ** 2)

    result = minimize_scalar(objective, bounds=(0.001, 1000), method="bounded")
    return result.x if result.success else np.nan


def compute_kendall_correlation(reference_counts, prediction_counts):
    """
    Compute Kendall’s tau using overlapping dates where both series are non-null.
    Returns NaN if there are fewer than 2 valid points or on error.
    """
    common_dates = set(reference_counts.keys()) & set(prediction_counts.keys())
    if not common_dates:
        return np.nan

    ref_values = []
    pred_values = []
    for d in common_dates:
        ref_val = reference_counts.get(d)
        pred_val = prediction_counts.get(d)
        if ref_val is not None and pred_val is not None:
            ref_values.append(ref_val)
            pred_values.append(pred_val)

    if len(ref_values) < 2 or len(pred_values) < 2:
        return np.nan

    try:
        kendall_corr, _ = kendalltau(ref_values, pred_values)
    except Exception as e:
        print(f"Kendall correlation error: {e}")
        kendall_corr = np.nan

    return kendall_corr


# Inputs: JSON files produced by 01_build_dict_inference.py
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

data = load_json_data(json_paths)

# Optional manual season definitions.
# For species with low counts, define a stable season per dataset/model here.
# If a custom season exists for (species, dataset, model), it overrides auto detection.
custom_season_dates = {
    "Fagus": {
        "2024_PBU": {
            "2025Q1_Beta_onnx_mixed_15sp": {"start": "2024-04-05", "end": "2024-05-19"},
            "2025Q2_Gamma_onnx_mixed_15sp_redFluo": {"start": "2024-04-05", "end": "2024-05-19"}
        },
        "2024_PBS": {
            "2025Q1_Beta_onnx_mixed_15sp": {"start": "2024-04-06", "end": "2024-05-05"},
            "2025Q2_Gamma_onnx_mixed_15sp_redFluo": {"start": "2024-04-06", "end": "2024-05-05"}
        },
        "2024_PLZ": {
            "2025Q1_Beta_onnx_mixed_15sp": {"start": "2024-04-06", "end": "2024-05-16"},
            "2025Q2_Gamma_onnx_mixed_15sp_redFluo": {"start": "2024-04-06", "end": "2024-05-16"}
        }, 
    },    
    "Quercus": {
        "2024_PBU": {
            "2025Q1_Beta_onnx_mixed_15sp": {"start": "2024-04-05", "end": "2024-05-17"},
            "2025Q2_Gamma_onnx_mixed_15sp_redFluo": {"start": "2024-04-05", "end": "2024-05-17"}
        },
        "2024_PBS": {
            "2025Q1_Beta_onnx_mixed_15sp": {"start": "2024-04-03", "end": "2024-05-05"},
            "2025Q2_Gamma_onnx_mixed_15sp_redFluo": {"start": "2024-04-03", "end": "2024-05-05"}
        },
        "2024_PLZ": {
            "2025Q1_Beta_onnx_mixed_15sp": {"start": "2024-04-07", "end": "2024-05-08"},
            "2025Q2_Gamma_onnx_mixed_15sp_redFluo": {"start": "2024-04-07", "end": "2024-05-08"}
        },
        "2024_PPY": {
            "2025Q1_Beta_onnx_mixed_15sp": {"start": "2024-05-03", "end": "2024-06-18"},
            "2025Q2_Gamma_onnx_mixed_15sp_redFluo": {"start": "2024-05-03", "end": "2024-06-18"}
        }
    },

    "Corylus": {
        "2024_PBU": {
            "2025Q1_Beta_onnx_mixed_15sp": {"start": "2024-01-23", "end": "2024-03-16"},
            "2025Q2_Gamma_onnx_mixed_15sp_redFluo": {"start": "2024-01-23", "end": "2024-03-16"}
        }
    },

    "Poaceae": {
        "2024_PLZ": {
            "2025Q1_Beta_onnx_mixed_15sp": {"start": "2024-04-05", "end": "2024-06-10"},
            "2025Q2_Gamma_onnx_mixed_15sp_redFluo": {"start": "2024-04-05", "end": "2024-06-10"}
        }
    }
}


rows = []

# Main loop over all combinations found in the merged JSON data
for model_name, model_data in data.items():
    for validation_dataset, dataset_data in model_data.items():
        for species, species_data in dataset_data.items():
            for threshold_str, species_threshold_data in species_data.items():
                # Threshold keys in the JSON are strings; convert for the table
                try:
                    threshold = float(threshold_str)
                except ValueError:
                    continue

                ref_counts = species_threshold_data.get('reference_counts', {})
                pred_counts = species_threshold_data.get('prediction_counts', {})

                if not ref_counts or not pred_counts:
                    print(f"Missing counts for {species} - {model_name} - {validation_dataset} - threshold {threshold}")
                    continue

                # Automatic season detection
                active_dates, season_start, season_end = find_season_active_dates(ref_counts)

                # Manual season override if provided
                if (
                    species in custom_season_dates
                    and validation_dataset in custom_season_dates[species]
                    and model_name in custom_season_dates[species][validation_dataset]
                ):
                    season_info = custom_season_dates[species][validation_dataset][model_name]
                    season_start = datetime.strptime(season_info["start"], "%Y-%m-%d")
                    season_end = datetime.strptime(season_info["end"], "%Y-%m-%d")
                    active_dates = {d for d in ref_counts if season_info["start"] <= d <= season_info["end"]}

                # Scaling factor based on in-season overlap
                scaling_factor = find_best_scaling_factor(ref_counts, pred_counts, active_dates)

                # Compute mean predicted counts inside and outside the season
                in_season_preds = [pred_counts[d] for d in pred_counts if d in active_dates and pred_counts[d] is not None]
                out_season_preds = [pred_counts[d] for d in pred_counts if d not in active_dates and pred_counts[d] is not None]

                mean_in_season = np.mean(in_season_preds) if in_season_preds else np.nan
                mean_out_season = np.mean(out_season_preds) if out_season_preds else np.nan

                if mean_in_season and mean_in_season > 0:
                    ratio_mean = mean_out_season / mean_in_season
                elif mean_out_season and mean_out_season > 0:
                    ratio_mean = mean_in_season / mean_out_season
                else:
                    ratio_mean = np.nan

                # Kendall correlation on overlapping non-null dates
                kendall_corr = compute_kendall_correlation(ref_counts, pred_counts)

                rows.append({
                    "species": species,
                    "model": model_name,
                    "dataset": validation_dataset,
                    "threshold": threshold,
                    "scaling_factor": scaling_factor,
                    "season_start": season_start.strftime("%Y-%m-%d") if season_start else "",
                    "season_end": season_end.strftime("%Y-%m-%d") if season_end else "",
                    "mean_in_season": mean_in_season,
                    "mean_out_season": mean_out_season,
                    "ratio_mean": ratio_mean,
                    "kendall_corr": kendall_corr
                })


# Build the final table
summary_df = pd.DataFrame(rows)

# Output location (directory is created if missing)
output_dir = "03_correlation/03.1_tables"
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(output_dir, "table_thresholds_corrK_SF_SigToNoise.csv")

# Save to CSV
summary_df.to_csv(output_csv_path, index=False)
print(f"Summary saved to: {output_csv_path}")
