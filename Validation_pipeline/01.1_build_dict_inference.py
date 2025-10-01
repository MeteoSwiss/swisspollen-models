"""
01_build_dict_inference.py
==========================

Purpose
-------
Build JSON files that hold daily reference (Hirst) and prediction counts
for multiple models, validation datasets, species, and confidence thresholds.

JSON structure
--------------
{
    "model_name": {
        "validation_dataset": {
            "species_1": {
                "confidence_threshold_1": {
                    "reference_counts": {  # Hirst counts per date
                        "date_1": count_value,
                        "date_2": count_value,
                        ...
                    },
                    "prediction_counts": {  # Model-inferred counts per date
                        "date_1": count_value,
                        "date_2": count_value,
                        ...
                    }
                },
                "confidence_threshold_2": { ... }
            },
            "species_2": { ... }
        },
        "validation_dataset_2": { ... }
    },
    "model_name_2": { ... }
}

Why these JSONs matter
----------------------
These JSON files are the foundation for the rest of the analysis.
Confidence thresholds are baked into the JSON. If you need a new threshold,
you must regenerate the JSONs here.

Inputs (configuration at the bottom of this file)
-------------------------------------------------
- CLASS_MAP:
  Dictionary where each key is the standardized Taxon name you want in the JSON,
  and the value is a list of class names (as found in the parquet predictions)
  that should be mapped/combined into that key. Keys should match column headers
  in the Hirst CSV. You can group multiple predicted classes into one key
  (e.g., "Pinus" and "Picea" grouped under "Pinaceae").

- data:
  - confidence_thresholds:
      Include 0 (needed by later scripts), then any additional thresholds you
      want to evaluate. Higher thresholds = fewer events retained.
  - joined_data_paths:
      List of folders with joined parquet outputs for each model/dataset.
  - csv_locations:
      List of paths to Hirst CSV files (reference counts).
      IMPORTANT: the order must match joined_data_paths and model_names.
  - model_names:
      Names of the models as used in the inference outputs. The order must match
      joined_data_paths and csv_locations.
  - save_path:
      Folder where JSONs will be written.

Outputs
-------
- /01_json_files/full_data
    JSONs with all available dates.
- /01_json_files/full_data
    JSONs with start/end trimmed (these are then used by the rest of the pipeline).

Notes and assumptions
---------------------
- Input paths need to be adapted to your filesystem structure. 

- Counts apply the "multiplier" found in the joined dataset. The device can subsample
  when event rates are high; the multiplier corrects counts accordingly.

- If a Hirst value is missing for a day, it is written as null in the JSON.
  Zeros in Hirst remain 0.

- If there are no predictions for a day but the day exists in the joined dataset,
  prediction count is 0. If the joined folder for that day is missing entirely,
  prediction count is null.

- This script can take time to run.

- There is also a helper that can trim dates from existing JSON files
  for selected validation datasets (see section "Trim the JSON to the dates you want").

Usage
-----
1) Set CLASS_MAP and the data configuration at the bottom.
2) Run the script. JSON files will be written to `save_path`.
3) Use the trimming section to drop unwanted dates from the JSONs.
4) run from the terminal as: python 01.1_build_dict_inference.py

"""

import pandas as pd
import pyarrow.dataset as ds
import json
import re
from typing import List, Dict, Optional
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta


def map_species_name(class_name: str, mapping: Dict[str, List[str]]) -> Optional[str]:
    """
    Map a raw class name (from predictions) to a standardized species name.

    - Matching is case-insensitive against all aliases provided in CLASS_MAP.
    - Returns the standardized species name (key in CLASS_MAP) with its original case.
    - Returns None if no match is found.
    """
    for species, aliases in mapping.items():
        if class_name.lower() in [alias.lower() for alias in aliases]:
            return species
    return None


def clean_nan_values(data_dict):
    """
    Replace NaN and Inf values with None so the result is valid JSON.

    Recursively walks nested dicts and lists.
    """
    for key, value in data_dict.items():
        if isinstance(value, dict):
            clean_nan_values(value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, float) and (np.isnan(item) or np.isinf(item)):
                    value[i] = None
        elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            data_dict[key] = None


def performance_analysis_with_counts(
    confidence_thresholds: List[float] = [0.99, 0.999, 0.9999],
    csv_locations: List[str] = None,
    joined_data_paths: List[str] = None,
    model_names: List[str] = None,
    save_path: Optional[str] = "output_timeseries_counts.json",
):
    """
    Build the JSON files containing daily reference and prediction counts for each:
    - model
    - validation dataset
    - species (mapped via CLASS_MAP)
    - confidence threshold

    Parameters must be aligned in order:
    csv_locations[i], joined_data_paths[i], model_names[i] refer to the same dataset.
    """
    if not (csv_locations and joined_data_paths and model_names):
        raise ValueError("CSV locations, joined data paths, and model names must be provided.")

    for csv_location, joined_data_path, model_name in zip(csv_locations, joined_data_paths, model_names):

        print(f"\nProcessing model: {model_name}")

        # Try to extract the validation dataset name from the folder path
        match = re.search(r'poleno-\d+/([^/]+)/joined', joined_data_path)
        validation_dataset_name = match.group(1) if match else "unknown_dataset"
        print(f"Validation dataset: {validation_dataset_name}")

        # Load Hirst reference data (daily counts). Assumes a 'Date' column.
        reference_df = pd.read_csv(csv_location)
        reference_df['Date'] = pd.to_datetime(reference_df['Date']).dt.strftime('%Y-%m-%d')
        reference_df.set_index('Date', inplace=True)

        # Standardize Hirst columns using the mapping (columns must match CLASS_MAP keys)
        reference_df.columns = [map_species_name(col, CLASS_MAP) for col in reference_df.columns]
        reference_df = reference_df.reset_index().rename(columns={'Date': 'date'})

        # Load parquet predictions (joined parquet dataset for this model/dataset)
        preds_dataset = ds.dataset(joined_data_path, exclude_invalid_files=True)

        # Collect all available dates present in the joined folder structure
        joined_data_root = Path(joined_data_path)
        available_dates_in_joined = set()

        for year_folder in joined_data_root.glob('*'):
            if not year_folder.is_dir():
                continue
            for month_folder in year_folder.glob('*'):
                if not month_folder.is_dir():
                    continue
                for day_folder in month_folder.glob('*'):
                    if day_folder.is_dir():
                        try:
                            date_str = f"{year_folder.name}-{month_folder.name}-{day_folder.name}"
                            pd.to_datetime(date_str)  # Validate date format
                            available_dates_in_joined.add(date_str)
                        except Exception:
                            pass

        # Columns expected in the parquet files for this model
        model_pred_col = f"{model_name}_0_preds"
        model_conf_col = f"{model_name}_0_confs"

        # Read necessary columns from parquet files
        scanner = preds_dataset.scanner(columns=["timestamp", model_pred_col, model_conf_col, "multiplier"])
        classifications_df = scanner.to_table().to_pandas()
        classifications_df['timestamp'] = pd.to_datetime(classifications_df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d')

        # If expected columns are missing, skip gracefully
        if model_pred_col not in classifications_df.columns or model_conf_col not in classifications_df.columns:
            print(f"Model {model_name} is missing expected columns.")
            continue

        # Overwrite with definitive list of available dates found in the parquet data
        available_dates_in_joined = set(classifications_df['timestamp'].dropna().unique())

        # Map raw predicted class names to standardized species names (CLASS_MAP)
        classifications_df["mapped_species"] = classifications_df[model_pred_col].apply(
            lambda x: map_species_name(x, CLASS_MAP)
        )

        # Prepare the output structure for this model/dataset
        model_timeseries_counts = {model_name: {validation_dataset_name: {}}}
        used_species = set()
        skipped_taxa = set()

        # Iterate over all raw taxa appearing in predictions
        for raw_taxa in classifications_df[model_pred_col].unique():
            mapped_species = map_species_name(raw_taxa, CLASS_MAP)
            if mapped_species is None:
                # Predicted class not mapped in CLASS_MAP -> not included in JSON
                skipped_taxa.add(raw_taxa)
                continue
            used_species.add(mapped_species)

            if mapped_species not in model_timeseries_counts[model_name][validation_dataset_name]:
                model_timeseries_counts[model_name][validation_dataset_name][mapped_species] = {}

            for confidence_threshold in confidence_thresholds:
                # Keep rows where confidence >= threshold and species matches
                filtered_df = classifications_df[
                    (classifications_df[model_conf_col] >= confidence_threshold)
                    & (classifications_df["mapped_species"] == mapped_species)
                ].copy()

                if filtered_df.empty:
                    print(f"No detections for {mapped_species} at threshold {confidence_threshold}. Filling with zeros or nulls.")

                # Ensure a daily date column
                filtered_df.loc[:, 'date'] = pd.to_datetime(filtered_df['timestamp']).dt.strftime('%Y-%m-%d')
                filtered_df.set_index('date', inplace=True)

                # For Raindrops/Water, ignore multiplier scaling
                if mapped_species == "Raindrops":
                    filtered_df['multiplier'] = 1

                # Compute daily counts using the multiplier and standard scale factor (57.6)
                daily_class_counts = filtered_df.groupby('date')['multiplier'].sum().reset_index(name='counts')
                daily_class_counts['counts'] = daily_class_counts['counts'] / 57.6

                # Build a complete daily index from the reference dates, then left-join predictions
                all_dates = pd.DataFrame({'date': reference_df['date'].astype(str)})
                daily_class_counts['date'] = daily_class_counts['date'].astype(str)
                daily_class_counts = all_dates.merge(daily_class_counts, on='date', how='left')

                # Build reference counts:
                # - If species is "Raindrops", reference is always 0.
                # - If the species column exists in the Hirst CSV, use it (sum if duplicated columns).
                # - Otherwise, reference is null for all dates.
                if mapped_species == "Raindrops":
                    reference_counts = {date: 0 for date in reference_df['date'].astype(str)}
                else:
                    ref_data = reference_df.set_index('date')

                    if mapped_species in ref_data.columns:
                        species_data = ref_data[mapped_species]

                        # If multiple columns share the same name, sum them row by row (keeping NaN rules)
                        if isinstance(species_data, pd.DataFrame):
                            species_data = species_data.sum(axis=1, min_count=1)

                        reference_counts = {
                            date: (None if pd.isna(value) else value)
                            for date, value in species_data.items()
                        }
                    else:
                        reference_counts = {date: None for date in reference_df['date'].astype(str)}

                # Build prediction counts per day:
                # - If the day folder is missing in the joined data, set to null.
                # - If the day exists but no events passed the threshold, set to 0.
                prediction_counts = {}
                for date in reference_df['date'].astype(str):
                    if date not in available_dates_in_joined:
                        prediction_counts[date] = None  # Missing day folder -> null
                    else:
                        value = daily_class_counts.set_index('date').loc[date, 'counts']
                        prediction_counts[date] = (0 if pd.isna(value) else value)

                # Store results for this species and threshold
                model_timeseries_counts[model_name][validation_dataset_name][mapped_species][str(confidence_threshold)] = {
                    "reference_counts": reference_counts,
                    "prediction_counts": prediction_counts,
                }

        # Report any taxa present in predictions but not mapped for JSON output
        skipped_taxa = skipped_taxa.difference(used_species)
        if skipped_taxa:
            print(f"Skipped taxa (present in parquet but not used in JSON): {', '.join(skipped_taxa)}")

        # Ensure JSON-safe values
        clean_nan_values(model_timeseries_counts)

        # Save one JSON per (model, validation_dataset)
        output_filename = f"{save_path}dict_{validation_dataset_name}_{model_name}.json"
        with open(output_filename, 'w') as f:
            json.dump(model_timeseries_counts, f, indent=4, sort_keys=True)

        print(f"Saved results to {output_filename}")

    return model_timeseries_counts


def remove_dates_from_json(
    file_paths: List[str],
    output_folder: str,
    dates_to_remove: List[str],
    target_validation_datasets: List[str]
):
    """
    Remove specific dates from reference_counts and prediction_counts in existing JSON files,
    but only for the listed validation datasets.

    This is useful to trim the time range used in later analyses.
    """
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)

        for model in data.values():
            for validation_dataset in target_validation_datasets:
                if validation_dataset in model:
                    for species in model[validation_dataset].values():
                        for threshold in species.values():
                            if "reference_counts" in threshold:
                                threshold["reference_counts"] = {
                                    date: count for date, count in threshold["reference_counts"].items()
                                    if date not in dates_to_remove
                                }
                            if "prediction_counts" in threshold:
                                threshold["prediction_counts"] = {
                                    date: count for date, count in threshold["prediction_counts"].items()
                                    if date not in dates_to_remove
                                }

        output_path = os.path.join(output_folder, os.path.basename(file_path))
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Processed and saved: {output_path}")


#########################################################  Modify after ##############################################################
"""
Configuration notes:

- CLASS_MAP:
    Dict mapping each JSON species name (key) to a list of aliases as they appear
    in parquet predictions. If several aliases map to the same key, their counts
    are combined.

- data:
    Contains the parameters passed into performance_analysis_with_counts:
        - confidence_thresholds: list of thresholds to compute.
        - joined_data_paths: paths to joined parquet folders for each dataset/model.
        - csv_locations: paths to the Hirst CSV files matching the same datasets.
        - model_names: model names matching the joined_data_paths (same order).
        - save_path: output folder for JSON files.

IMPORTANT: Order matters.
For each index i, joined_data_paths[i], csv_locations[i], and model_names[i]
must all refer to the same dataset and model.
"""
# Define species class mapping
# If multiple taxa are listed under a key, they are combined in the JSON.
CLASS_MAP = {
    "Alnus": ["Alnus", "alnus"],
    "Corylus": ["Corylus", "corylus"],
    "Pinaceae": ["Pinaceae", "pinaceae", "Pinus", "pinus", "Picea", "picea"],
    "Fraxinus": ["Fraxinus", "fraxinus"],
    "Raindrops": ["Raindrops", "water"],
    "Betula": ["Betula", "betula"],
    "Carpinus": ["Carpinus", "carpinus"],
    "Fagus": ["Fagus", "fagus"],
    "Populus": ["Populus", "populus"],
    "Quercus": ["Quercus", "quercus"],
    "Taxus": ["Taxus", "taxus"],
    "Ulmus": ["Ulmus", "ulmus"],
    "Poaceae": ["Poaceae", "poaceae"],
    "Platanus": ["Platanus", "platanus"],
    "Cupressaceae": ["Cupressaceae", "cupressus"],
}

# NOTE ON ORDER:
# Keep joined_data_paths, csv_locations, and model_names aligned by position.
# This is how the code knows which files belong together.

data = performance_analysis_with_counts(
    confidence_thresholds=[0, 0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999],

    joined_data_paths=[
        '/inference_output/model2025/data/poleno-13/2024_PBU/joined/2025Q1_Beta_onnx_mixed_15sp',
        '/inference_output/model2025/data/poleno-27/2024_PBS/joined/2025Q1_Beta_onnx_mixed_15sp',
        '/inference_output/model2025/data/poleno-30/2024_PPY/joined/2025Q1_Beta_onnx_mixed_15sp',
        '/inference_output/model2025/data/poleno-18/2024_PNE/joined/2025Q1_Beta_onnx_mixed_15sp',
        '/inference_output/model2025/data/poleno-21/2024_PLZ/joined/2025Q1_Beta_onnx_mixed_15sp',
        '/inference_output/model2025/data/poleno-13/2024_PBU/joined/2025Q2_Gamma_onnx_mixed_15sp_redFluo',
        '/inference_output/model2025/data/poleno-27/2024_PBS/joined/2025Q2_Gamma_onnx_mixed_15sp_redFluo',
        '/inference_output/model2025/data/poleno-30/2024_PPY/joined/2025Q2_Gamma_onnx_mixed_15sp_redFluo',
        '/inference_output/model2025/data/poleno-18/2024_PNE/joined/2025Q2_Gamma_onnx_mixed_15sp_redFluo',
        '/inference_output/model2025/data/poleno-21/2024_PLZ/joined/2025Q2_Gamma_onnx_mixed_15sp_redFluo',
            ],

    csv_locations=[
        '2024_PBU_Hirst.csv',
        '2024_PBS_Hirst.csv',
        '2024_PPY_Hirst.csv',
        '2024_PNE_Hirst.csv', 
        '2024_PLZ_Hirst.csv',
        '2024_PBU_Hirst.csv',
        '2024_PBS_Hirst.csv',
        '2024_PPY_Hirst.csv',
        '2024_PNE_Hirst.csv', 
        '2024_PLZ_Hirst.csv'      
    ],
    model_names=[
        '2025Q1_Beta_onnx_mixed_15sp',
        '2025Q1_Beta_onnx_mixed_15sp',
        '2025Q1_Beta_onnx_mixed_15sp',
        '2025Q1_Beta_onnx_mixed_15sp',
        '2025Q1_Beta_onnx_mixed_15sp',
        '2025Q2_Gamma_onnx_mixed_15sp_redFluo',
        '2025Q2_Gamma_onnx_mixed_15sp_redFluo',
        '2025Q2_Gamma_onnx_mixed_15sp_redFluo',
        '2025Q2_Gamma_onnx_mixed_15sp_redFluo',
        '2025Q2_Gamma_onnx_mixed_15sp_redFluo'
        ],
    save_path='01_json_files/full_data/'
)

####### Trim the JSON to the dates you want

# Folder that contains the JSON files created above
input_folder = "01_json_files/full_data"
output_folder = "01_json_files"

# Collect all JSON file paths
file_paths = [
    os.path.join(input_folder, f)
    for f in os.listdir(input_folder)
    if f.endswith(".json")
]

# Validation datasets to modify in the trimming step
target_validation_datasets = ["2024_PBU", "2024_PBS", "2024_PNE", "2024_PLZ", "2024_PPY"]

# Example: remove dates between 2024-09-15 and 2024-12-31 (inclusive)
dates_to_remove = [
    (datetime(2024, 9, 15) + timedelta(days=i)).strftime("%Y-%m-%d")
    for i in range((datetime(2024, 12, 31) - datetime(2024, 9, 15)).days + 1)
]

# Apply the trimming to all JSONs in 'file_paths'
remove_dates_from_json(file_paths, output_folder, dates_to_remove, target_validation_datasets)

print("All files processed successfully.")
