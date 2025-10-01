"""
03.1_correlation.py
===============================

Purpose
-------
Produce a table of correlation coefficients (Kendall, Spearman, Pearson) between
model prediction counts and Hirst reference counts, for each species, model, and
validation dataset, across multiple confidence thresholds.


Inputs
------
- A list of JSON files produced by 01_build_dict_inference.py.
  Each JSON has the structure:
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

Outputs
-------
- One CSV with, for each (species, model, dataset), and for each threshold:
    * <thr>_kendall, <thr>_spearman, <thr>_pearson
    * <thr>_common_dates  (number of dates with both ref and pred values present)
    * <thr>_ref_mean, <thr>_ref_std, <thr>_ref_min, <thr>_ref_max, <thr>_ref_unique_values

  Example destination:
    /scratch/mmp/ml_workflow/correlation/03_correlation/03.1_tables/correlation_table.csv

What the script does (plain English)
------------------------------------
1) load_json_data(json_paths)
   - Reads and merges many per-model JSON files into a single nested dictionary,
     without overwriting existing date entries.

2) calculate_correlations(reference_counts, prediction_counts)
   - Aligns on common dates, drops nulls, and computes Kendall, Spearman, Pearson.
   - Returns NaN when data are insufficient (e.g., <2 points) or constant.

3) generate_correlation_table(data, thresholds, species_list)
   - For each (model, dataset, species), and for each threshold:
     * Fetches the time series
     * Computes correlations + simple reference statistics
     * Produces one row per (species, model, dataset)

4) Writes the final CSV to disk.

Notes
-----
- Directory creation: make sure the output folder exists before running.
- Threshold "0" (baseline) should be present in the JSONs.
- Correlations are computed only on dates where both ref and pred are available
  and non-null. If there are <2 valid points, coefficients are set to NaN.
- Constant series (all same value) also yield NaN correlations by design.

Quick start
-----------
1) Adjust:
   - species_list
   - thresholds
   - json_paths
   - save_dir

2) Install dependencies:
   pip install pandas numpy plotly scipy

3) Run:
   python 03.1_build_correlation_table.py
"""

import plotly.graph_objects as go
import pandas as pd
import os
import json
import plotly.graph_objects as go
import plotly.express as px  # Import plotly express to access color scales
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr


# Utility function to load data from multiple JSON files
# Utility function to load data from multiple JSON files
def load_json_data(json_paths):
    """
    Read and combine multiple JSON files from 01_build_dict_inference.py
    into a single nested dictionary. When the same (model/dataset/species/threshold)
    appears more than once, new date->count pairs are added without overwriting
    existing entries.
    """
    combined_data = {}
    
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
            # Iterate over the loaded data
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
                                        # Extract the existing counts (if any)
                                        existing_counts = combined_data[model_name][validation_dataset_name][species][threshold].get(key, {})
                                        new_counts = threshold_data.get(key, {})

                                        # Add the new date/count pair only if it doesn't already exist
                                        for date, count in new_counts.items():
                                            if date not in existing_counts:
                                                existing_counts[date] = count  # Add the new date/count
                                                
                                        # Update the dictionary with the merged counts
                                        combined_data[model_name][validation_dataset_name][species][threshold][key] = existing_counts

    return combined_data


def calculate_correlations(reference_counts, prediction_counts):
    """
    Compute Kendall, Spearman, and Pearson correlations between reference and
    prediction daily counts on the intersection of their dates.

    Returns a dict with the three coefficients and simple reference stats.
    If there are fewer than 2 valid points, or if one series is constant,
    returns NaN for the coefficients.
    """
    # Strip time part from the reference and prediction dates (only keep 'YYYY-MM-DD')
    reference_counts_cleaned = {date.split(' ')[0]: count for date, count in reference_counts.items()}
    prediction_counts_cleaned = {date.split(' ')[0]: count for date, count in prediction_counts.items()}
    
    # Find common dates
    common_dates = set(reference_counts_cleaned.keys()) & set(prediction_counts_cleaned.keys())
    sorted_dates = sorted(common_dates)

    # Extract values only for common dates, skipping nulls
    ref_values = []
    pred_values = []
    for date in sorted_dates:
        ref_value = reference_counts_cleaned.get(date)
        pred_value = prediction_counts_cleaned.get(date)

        if ref_value is not None and pred_value is not None:
            ref_values.append(ref_value)
            pred_values.append(pred_value)

    # If there are no valid values after filtering, return NaNs
    if not ref_values or not pred_values:
        return {
            "kendall": np.nan, "spearman": np.nan, "pearson": np.nan,
            "common_dates_count": len(common_dates),
            "ref_mean": np.nan, "ref_std": np.nan, 
            "ref_min": np.nan, "ref_max": np.nan, "ref_unique_values": np.nan
        }

    # Compute reference statistics
    ref_mean = np.mean(ref_values)
    ref_std = np.std(ref_values, ddof=1) if len(ref_values) > 1 else np.nan
    ref_min = np.min(ref_values) if ref_values else np.nan
    ref_max = np.max(ref_values) if ref_values else np.nan
    ref_unique_values = len(set(ref_values)) if ref_values else np.nan

    common_dates_count = len(common_dates)

    # If not enough data (less than 2 common dates), return NaNs
    if len(ref_values) < 2:
        return {
            "kendall": np.nan, "spearman": np.nan, "pearson": np.nan,
            "common_dates_count": common_dates_count,
            "ref_mean": ref_mean, "ref_std": ref_std, 
            "ref_min": ref_min, "ref_max": ref_max, "ref_unique_values": ref_unique_values
        }

    # Check for constant data (all values the same for reference or prediction)
    if len(set(ref_values)) == 1 or len(set(pred_values)) == 1:
        return {
            "kendall": np.nan, "spearman": np.nan, "pearson": np.nan,
            "common_dates_count": common_dates_count,
            "ref_mean": ref_mean, "ref_std": ref_std, 
            "ref_min": ref_min, "ref_max": ref_max, "ref_unique_values": ref_unique_values
        }

    # Compute correlations
    kendall_corr, _ = kendalltau(ref_values, pred_values)
    spearman_corr, _ = spearmanr(ref_values, pred_values)
    pearson_corr, _ = pearsonr(ref_values, pred_values)

    return {
        "kendall": kendall_corr, "spearman": spearman_corr, "pearson": pearson_corr,
        "common_dates_count": common_dates_count,
        "ref_mean": ref_mean, "ref_std": ref_std, 
        "ref_min": ref_min, "ref_max": ref_max, "ref_unique_values": ref_unique_values
    }


def generate_correlation_table(data, thresholds, species_list):
    """
    Build a dataframe with one row per (species, model, dataset), containing
    correlation metrics for each threshold requested.

    For each threshold T in 'thresholds', the following columns are added:
      T_kendall, T_spearman, T_pearson,
      T_common_dates, T_ref_mean, T_ref_std, T_ref_min, T_ref_max, T_ref_unique_values
    """
    table_rows = []

    for model, model_data in data.items():
        for validation_dataset, val_data in model_data.items():
            for species in species_list:
                species_data = val_data.get(species, {})
                species_results = {"species": species, "model": model, "dataset": validation_dataset}

                for threshold in thresholds:
                    str_threshold = str(threshold)
                    reference_counts = species_data.get(str_threshold, {}).get('reference_counts', {})
                    prediction_counts = species_data.get(str_threshold, {}).get('prediction_counts', {})

                    # If either series is missing, fill with NaNs/zeros as appropriate
                    if not reference_counts or not prediction_counts:
                        species_results[f"{threshold}_kendall"] = np.nan
                        species_results[f"{threshold}_spearman"] = np.nan
                        species_results[f"{threshold}_pearson"] = np.nan
                        species_results[f"{threshold}_common_dates"] = 0
                        species_results[f"{threshold}_ref_mean"] = np.nan
                        species_results[f"{threshold}_ref_std"] = np.nan
                        species_results[f"{threshold}_ref_min"] = np.nan
                        species_results[f"{threshold}_ref_max"] = np.nan
                        species_results[f"{threshold}_ref_unique_values"] = np.nan
                    else:
                        # Compute correlations and reference stats
                        correlations = calculate_correlations(reference_counts, prediction_counts)
                        species_results[f"{threshold}_kendall"] = correlations["kendall"]
                        species_results[f"{threshold}_spearman"] = correlations["spearman"]
                        species_results[f"{threshold}_pearson"] = correlations["pearson"]
                        species_results[f"{threshold}_common_dates"] = correlations["common_dates_count"]
                        species_results[f"{threshold}_ref_mean"] = correlations["ref_mean"]
                        species_results[f"{threshold}_ref_std"] = correlations["ref_std"]
                        species_results[f"{threshold}_ref_min"] = correlations["ref_min"]
                        species_results[f"{threshold}_ref_max"] = correlations["ref_max"]
                        species_results[f"{threshold}_ref_unique_values"] = correlations["ref_unique_values"]

                table_rows.append(species_results)

    return pd.DataFrame(table_rows)


# Example usage:
# Define species, thresholds, and your data
species_list = ["Poaceae", "Platanus", "Alnus", "Betula", "Carpinus", "Corylus", "Cupressaceae", "Fagus", "Fraxinus", "Pinaceae", "Populus", "Quercus", "Taxus", "Ulmus", "Raindrops", "Juglans", "Moraceae", "Rumex", "Sambucus"]
thresholds=[0, 0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]

# JSONs from 01_build_dict_inference.py
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

# Load all JSONs into a combined structure
data = load_json_data(json_paths)

# Directory to save the CSV table (ensure it exists beforehand)
save_dir = '03_correlation/03.1_tables/'

# Generate the correlation table
correlation_df = generate_correlation_table(data, thresholds, species_list)

# Save the table to a CSV file
correlation_df.to_csv(f"{save_dir}correlation_table.csv", index=False)
