"""
02.3_low_count_validation_datasets.py
================================

Purpose
-------
Give a quick overview of which validation datasets have high or low Hirst
reference counts. This helps decide which correlation coefficients to trust:
we should not rely on correlations computed from time series with consistently
low reference counts.

Inputs
------
- A list of JSON files produced by 01_build_dict_inference.py.
  Each JSON contains, for every model / validation dataset / species / threshold:
    * reference_counts: { "YYYY-MM-DD": number | null }
    * prediction_counts: { "YYYY-MM-DD": number | null }

Outputs
-------
- One CSV summarizing, for each (model, validation dataset, species) and for
  every available threshold in the JSON, the min/max of reference and prediction
  counts observed in the time series.

  Example destination:
  02_data_discard/02.3_counts/low_high_counts.csv

How it works (plain English)
----------------------------
- For each JSON file:
  - Loop over models, then validation datasets, then species.
  - For every threshold found in that species:
    * Scan the daily "reference_counts" and "prediction_counts".
    * Compute min and max values, ignoring nulls.
    * Store four numbers per threshold:
        <threshold>_ref_min, <threshold>_ref_max,
        <threshold>_pred_min, <threshold>_pred_max.
- Merge results across all JSONs and write one CSV.

Notes
-----
- If a series has no valid (non-null) values for a given threshold, its min/max
  is written as 0.
- This script does not trim dates or define a period; any time-range selection
  must be done upstream when generating or trimming the JSONs.
- Ensure the output folder exists before running (the script does not create it).
- No external packages required (uses only the Python standard library).
"""

import json
import csv

'''
This code aims at giving an overview of what validation datasets show high reference counts or low reference counts. 
Aims at deciding which correlation coefficients should be "trusted" more than others, 
as well as for calculating scaling factors: we don't want them to be calculated from timeseries with low reference counts, because of the lack of accuracy.
'''

# Function to process a single JSON file dynamically for all thresholds
def process_json_data(json_data):
    """
    Read one JSON structure (from 01_build_dict_inference.py) and collect, for each
    model / validation dataset / species / threshold:
      - min and max of reference_counts (ignoring nulls)
      - min and max of prediction_counts (ignoring nulls)

    Returns a list of dictionaries, each row starting with:
        model, species, validation_dataset
    followed by:
        <threshold>_ref_min, <threshold>_ref_max,
        <threshold>_pred_min, <threshold>_pred_max
    for all thresholds present for that species.
    """
    results = []
    
    # Iterate over each model in the JSON data
    for model_name, model_data in json_data.items():
        
        # Iterate over each validation dataset for the current model
        for dataset_name, dataset_data in model_data.items():
            
            # Iterate over each species in the current validation dataset
            for species_name, species_data in dataset_data.items():
                
                # This row will accumulate min/max per threshold for this species
                threshold_results = {
                    "model": model_name,
                    "species": species_name,
                    "validation_dataset": dataset_name
                }

                # Process each threshold dynamically (keys inside species_data)
                for threshold_name, threshold_data in species_data.items():
                    reference_counts = threshold_data["reference_counts"]
                    prediction_counts = threshold_data["prediction_counts"]

                    # Start with "no data" sentinels
                    min_ref, max_ref = float('inf'), float('-inf')
                    min_pred, max_pred = float('inf'), float('-inf')

                    # Scan all dates in reference counts, skip nulls
                    for date_key, reference_count in reference_counts.items():
                        if reference_count is not None:
                            min_ref = min(min_ref, reference_count)
                            max_ref = max(max_ref, reference_count)
                    
                    # Scan all dates in prediction counts, skip nulls
                    for date_key, prediction_count in prediction_counts.items():
                        if prediction_count is not None:
                            min_pred = min(min_pred, prediction_count)
                            max_pred = max(max_pred, prediction_count)

                    # If no valid values were found, write 0
                    threshold_results[f"{threshold_name}_ref_min"] = min_ref if min_ref != float('inf') else 0
                    threshold_results[f"{threshold_name}_ref_max"] = max_ref if max_ref != float('-inf') else 0
                    threshold_results[f"{threshold_name}_pred_min"] = min_pred if min_pred != float('inf') else 0
                    threshold_results[f"{threshold_name}_pred_max"] = max_pred if max_pred != float('-inf') else 0

                results.append(threshold_results)

    return results


# Function to process multiple JSON files and write the results to a CSV file
def process_multiple_json_files(json_files, output_csv):
    """
    Read all listed JSON files, aggregate the min/max stats produced by
    process_json_data, and write one CSV file.

    The header includes:
      - model, species, validation_dataset
      - all threshold-based columns (collected dynamically)
    """
    all_results = []

    # Iterate over each JSON file and collect rows
    for json_file in json_files:
        with open(json_file, 'r') as file:
            json_data = json.load(file)
            file_results = process_json_data(json_data)
            all_results.extend(file_results)

    # Collect all unique column names that appear across rows
    threshold_columns = set()
    for result in all_results:
        threshold_columns.update(result.keys())

    # Order columns: identifiers first, then the dynamic threshold columns
    threshold_columns = sorted(threshold_columns)

    # Write the results to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['model', 'species', 'validation_dataset'] + [
            col for col in threshold_columns if col not in ['model', 'species', 'validation_dataset']
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Results written to {output_csv}")


# List of JSON files produced by 01_build_dict_inference.py
json_files = [
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

# Where to write the CSV.
# Make sure the folder exists before running (the script does not create directories).
output_csv = "02_data_discard/02.3_counts/low_high_counts.csv"

# Run the processing
process_multiple_json_files(json_files, output_csv)
