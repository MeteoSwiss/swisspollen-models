"""
02.1_Plots_validation.py
====================================

Quick start (CLI)
-----------------
1) Create and activate a virtual env, then install deps:
   python -m venv .venv && . .venv/bin/activate
   pip install -r requirements.txt
   # (Windows PowerShell) py -m venv .venv; .\.venv\Scripts\Activate.ps1

2) Point 'json_paths' to the JSON files produced by 01_build_dict_inference.py,
   and set 'save_dir' paths at the bottom of this file.

3) Run:
   python 02_threshold_plots_and_timeseries.py

What this script does
---------------------
This script loads the JSON outputs from 01_build_dict_inference.py and produces:
- Interactive HTML plots showing, by threshold, how much data is discarded
  (per validation dataset, species, and model).
- Interactive HTML time series (reference vs predictions at multiple thresholds).
- CSV summaries of the threshold that still keeps ≥ X% of events (default 70%),
  per model / validation dataset / species.
- A CSV of the percentage retained at each threshold for all models/species.
- (Optional) A combined JSON merging many per-model JSON files (for convenience).

Why you need to run it
----------------------
Later scripts rely on:
- Per-model CSV of the "max threshold keeping ≥ 70%".
- The "all_models_threshold_retention.csv" summary.
- The threshold discard HTML plots for QA/QC.

Functions overview
------------------
load_json_data(json_paths)
    Read multiple JSON files from 01_build_dict_inference.py and merge them into
    a single nested dict without overwriting existing dates.

plot_threshold_discard(data, thresholds, save_path)
    For each validation dataset, plot an interactive HTML figure showing
    the percentage of events removed as the threshold increases. One color per species,
    one line style per model. Includes date range in the title.

save_threshold_discard_table(data, save_path, model_name=None, keep_percentage=70)
    For each species and validation dataset, find the highest threshold at which at
    least 'keep_percentage' of event-days are retained (baseline = threshold 0).
    Saves one CSV per model.

plot_timeseries(data, thresholds, save_dir)
    For each model / validation dataset / species, plot reference (Hirst) vs
    predictions for all provided thresholds to an interactive HTML timeseries.

plot_timeseries_for_selected_species_and_models(data, species_thresholds, validation_datasets, models, save_dir)
    Fine-grained plotting: pick specific species with their own threshold lists,
    select the models and validation datasets, and generate a combined interactive plot.

save_all_threshold_retention_tables(data, save_path)
    Produce 'all_models_threshold_retention.csv' with the % kept at every threshold
    for all model/dataset/species combinations.

Inputs (you edit these at the bottom)
-------------------------------------
- json_paths: list of JSON files created by 01_build_dict_inference.py
- thresholds: the thresholds you want drawn in timeseries/retention plots
- save_dir / save_path: folders where HTML/CSV outputs are written

Outputs
-------
Folders (examples you use below):
-   * <dataset>_removed_proportions_by_threshold.html
    * <model>_thresholds_to_keep_70pct.csv
    * all_models_threshold_retention.csv
- /02_data_discard/02.1_timeseries/
    * <model>_<dataset>_<species>_timeseries.html
- /02_data_discard/02.1_timeseries/combinations/
    * <datasets>_<models>_<species>_timeseries.html

Conventions and notes
---------------------
- "Threshold 0" is the baseline (expected to be present) and represents "100% kept".
- "Percentage kept" is computed from the number of *non-zero* daily predictions,
  relative to baseline (threshold 0). If total_count at baseline is zero, that
  model/species/dataset is skipped.
- Null vs 0: Null means no data for that date (e.g., missing joined folder).
  Zero means the date exists but no events passed the threshold.
- You can safely run this on subsets first to test file paths and plot shapes.
"""

import plotly.graph_objects as go
import pandas as pd
import os
import json
import plotly.graph_objects as go
import plotly.express as px  # Used only to obtain color scales
import numpy as np
import csv
from datetime import datetime


# Utility: merge multiple per-model JSONs into one nested dict
def load_json_data(json_paths):
    """
    Read and merge JSON files produced by 01_build_dict_inference.py.

    Merging behavior:
    - Models, datasets, species, thresholds are combined.
    - For each (reference_counts|prediction_counts), new date->count pairs are added
      without overwriting any existing date already present in the combined dict.
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
                            # First time we see this species -> copy everything
                            combined_data[model_name][validation_dataset_name][species] = species_data
                        else:
                            # Merge thresholds for an already-seen species
                            for threshold, threshold_data in species_data.items():
                                if threshold not in combined_data[model_name][validation_dataset_name][species]:
                                    combined_data[model_name][validation_dataset_name][species][threshold] = threshold_data
                                else:
                                    for key in ['reference_counts', 'prediction_counts']:
                                        # Existing date->count pairs (if any)
                                        existing_counts = combined_data[model_name][validation_dataset_name][species][threshold].get(key, {})
                                        new_counts = threshold_data.get(key, {})

                                        # Add only dates that are not already present
                                        for date, count in new_counts.items():
                                            if date not in existing_counts:
                                                existing_counts[date] = count
                                                
                                        combined_data[model_name][validation_dataset_name][species][threshold][key] = existing_counts

    return combined_data


def plot_threshold_discard(data, thresholds, save_path):
    """
    For each validation dataset:
    - Plot % of events removed vs threshold (one trace per species x model).
    - Color encodes species; line style encodes model.
    - Title also shows min/max dates present at baseline (threshold == 0).
    Saves one HTML file per validation dataset.
    """
    # Different line styles distinguish models
    line_styles = ['solid', 'dash', 'dot', 'dashdot']

    # Fixed color list for species (extended as needed)
    species_colors = [
        'blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow'
    ]

    # Iterate over validation datasets (assumes same set of datasets across models)
    for validation_dataset_name in next(iter(data.values())).keys():
        fig = go.Figure()

        # Track the global date range for this dataset (based on threshold 0)
        min_date = None
        max_date = None

        # Iterate over models
        for i, (model_name, model_data) in enumerate(data.items()):
            if not model_data:
                print(f"No data found for model: {model_name}. Skipping...")
                continue

            # Data for this validation dataset
            validation_data = model_data.get(validation_dataset_name, {})

            # Iterate over species
            for j, (species, species_data) in enumerate(validation_data.items()):
                initial_sum = None
                percentage_discarded = []

                # Iterate over thresholds
                for threshold in thresholds:
                    str_threshold = str(threshold)

                    # Daily predictions at this threshold (dates are keys)
                    prediction_counts = species_data.get(str_threshold, {}).get('prediction_counts', {})

                    # Update min/max dates when threshold == 0 (baseline)
                    if threshold == 0:
                        dates_at_threshold_0 = prediction_counts.keys()
                        if dates_at_threshold_0:
                            dates_at_threshold_0 = [datetime.strptime(date, "%Y-%m-%d") for date in dates_at_threshold_0]
                            current_min_date = min(dates_at_threshold_0)
                            current_max_date = max(dates_at_threshold_0)
                            if min_date is None or current_min_date < min_date:
                                min_date = current_min_date
                            if max_date is None or current_max_date > max_date:
                                max_date = current_max_date

                    # Baseline total (100% kept). Uses sum of non-null values at this threshold.
                    if initial_sum is None:
                        initial_sum = sum(v for v in prediction_counts.values() if v is not None)

                    # Sum values at this threshold (only non-null)
                    sum_at_threshold = sum(v for v in prediction_counts.values() if v is not None and v >= threshold)

                    # % discarded relative to baseline
                    percentage_discarded.append(100 - (sum_at_threshold / initial_sum) * 100 if initial_sum else 0)

                # Add one trace per species x model
                fig.add_trace(go.Scatter(
                    x=thresholds,
                    y=percentage_discarded,
                    mode='lines+markers',
                    name=f'{model_name} - {species}',
                    line=dict(dash=line_styles[i % len(line_styles)]),  # style by model
                    marker=dict(color=species_colors[j % len(species_colors)])  # color by species
                ))

        # Title with date range
        date_range_str = ""
        if min_date and max_date:
            date_range_str = f"Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"

        fig.update_layout(
            title=f"Percentage of Data Removed by Threshold - {validation_dataset_name} ({date_range_str})",
            xaxis_title="Threshold",
            yaxis_title="% of Events Removed",
            showlegend=True,
            template="plotly_white"
        )

        # Write one HTML per validation dataset
        os.makedirs(save_path, exist_ok=True)
        validation_save_path = os.path.join(save_path, f'{validation_dataset_name}_removed_proportions_by_threshold.html')
        fig.write_html(validation_save_path)
        print(f"Saved plot for validation dataset {validation_dataset_name} at {validation_save_path}")


def save_threshold_discard_table(data, save_path, model_name=None, keep_percentage=70):
    """
    For a given model (or for all models if model_name is None), find—per species and validation dataset—
    the highest threshold at which at least 'keep_percentage'% of event-days remain
    (baseline = threshold '0', counting days with non-zero predictions).

    Writes one CSV per model to 'save_path'.
    """
    os.makedirs(save_path, exist_ok=True)

    # Limit to one model if requested
    if model_name:
        data = {model_name: data.get(model_name, {})}

    # Iterate over models
    for model_name, model_data in data.items():
        if not model_data:
            print(f"No data found for model: {model_name}. Skipping...")
            continue

        table_rows = []

        # Per validation dataset
        for validation_dataset_name, validation_data in model_data.items():
            # Per species
            for species, species_data in validation_data.items():
                threshold_to_keep = None

                # Collect and sort thresholds as numbers (keys are strings)
                available_thresholds = sorted(
                    [float(threshold) for threshold in species_data.keys() if threshold != 'prediction_counts']
                )
                print(f"Available thresholds for {species} in {validation_dataset_name}: {available_thresholds}")

                if not available_thresholds:
                    print(f"No thresholds found for species {species} in validation dataset {validation_dataset_name}")
                    continue

                baseline_threshold = '0'  # always present

                # Count event-days with non-zero predictions at baseline
                total_count = sum(
                    1 for v in species_data.get(baseline_threshold, {}).get('prediction_counts', {}).values()
                    if v is not None and v > 0
                )
                print(f"The total count for {species} in {validation_dataset_name} at threshold 0.0 is {total_count}")

                # Walk thresholds in ascending order; keep raising while ≥ keep_percentage
                for threshold in available_thresholds:
                    str_threshold = str(threshold)
                    print(f"Checking threshold {str_threshold} for species {species}...")

                    prediction_counts = species_data.get(str_threshold, {}).get('prediction_counts', {})
                    non_zero_count = sum(1 for v in prediction_counts.values() if v is not None and v > 0)
                    print(f"Non-zero entries at threshold {threshold}: {non_zero_count}")

                    if non_zero_count == 0:
                        print(f"{str_threshold} has no valid prediction counts for {species}. Skipping...")
                        continue

                    percentage_kept = (non_zero_count / total_count) * 100 if total_count else 0
                    print(f"Percentage kept at threshold {threshold}: {percentage_kept:.2f}%")

                    if percentage_kept >= keep_percentage:
                        threshold_to_keep = threshold
                    else:
                        # thresholds are sorted; once we drop below target, stop
                        break

                if threshold_to_keep is not None:
                    table_rows.append({
                        'Model Name': model_name,
                        'Validation Dataset': validation_dataset_name,
                        'Species': species,
                        'Threshold': threshold_to_keep
                    })

        # Save one CSV per model
        if table_rows:
            model_save_path = os.path.join(save_path, f'{model_name}_thresholds_to_keep_{keep_percentage}pct.csv')
            fieldnames = ['Model Name', 'Validation Dataset', 'Species', 'Threshold']
            with open(model_save_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(table_rows)
            print(f"Saved threshold discard table for model {model_name} at {model_save_path}")
        else:
            print(f"No thresholds found for model {model_name}. No table saved.")


def plot_timeseries(data, thresholds, save_dir):
    """
    For each model / validation dataset / species:
    - Plot reference (Hirst) once (from threshold 0).
    - Plot predictions for every threshold in 'thresholds'.
    - Save one HTML timeseries per (model, dataset, species).
    """
    # Build a palette from a Plotly sequential colorscale sized to #thresholds
    colorscale = px.colors.sequential.Viridis
    num_thresholds = len(thresholds)
    color_map = (colorscale * (num_thresholds // len(colorscale) + 1))[:num_thresholds]

    # Iterate through models
    for model_name, model_data in data.items():
        # Iterate through validation datasets
        for val_dataset, val_data in model_data.items():
            # Iterate through species
            for species, species_data in val_data.items():
                fig = go.Figure()
                reference_plotted = False

                # For every threshold
                for idx, threshold in enumerate(thresholds):
                    reference_counts = species_data.get(str(threshold), {}).get('reference_counts', {})
                    prediction_counts = species_data.get(str(threshold), {}).get('prediction_counts', {})
                    
                    # Make a single sorted date axis
                    dates = sorted(set(reference_counts.keys()) | set(prediction_counts.keys()))
                    dates = [date.split(' ')[0] for date in dates]

                    ref_values = [reference_counts.get(date, 0) for date in dates]
                    pred_values = [prediction_counts.get(date, 0) for date in dates]

                    # Plot reference once
                    if not reference_plotted:
                        fig.add_trace(go.Scatter(
                            x=dates, 
                            y=ref_values, 
                            mode='lines', 
                            name=f'{model_name} - {val_dataset} - {species} - Reference', 
                            line=dict(dash='solid', color='blue')
                        ))
                        reference_plotted = True

                    # Plot predictions for each threshold
                    fig.add_trace(go.Scatter(
                        x=dates, 
                        y=pred_values, 
                        mode='lines', 
                        name=f'{model_name} - {val_dataset} - {species} - Threshold {threshold}', 
                        line=dict(dash='solid', color=color_map[idx])
                    ))

                fig.update_layout(
                    title=f"{model_name} - {val_dataset} - {species} - Threshold Timeseries",
                    xaxis_title="Date",
                    yaxis_title="Count",
                    showlegend=True,
                    template="plotly_white"
                )

                # Save HTML
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{model_name}_{val_dataset}_{species}_timeseries.html")
                fig.write_html(save_path)


def plot_timeseries_for_selected_species_and_models(data, species_thresholds, validation_datasets, models, save_dir):
    """
    Flexible plotter to compare selected species (with custom threshold lists)
    across chosen models and validation datasets, all in one interactive HTML plot.

    Notes:
    - Include threshold 0 in 'species_thresholds' for a reference line.
    - The same threshold list is applied to a species across all selected models/datasets.
    """
    # Build a large qualitative palette
    colorscale = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Dark2

    num_colors_needed = sum(len(thresholds) for thresholds in species_thresholds.values()) + len(species_thresholds)
    colors = [colorscale[i % len(colorscale)] for i in range(num_colors_needed)]

    fig = go.Figure()

    # Track global date range
    all_dates = set()
    color_index = 0

    # Iterate selected models and datasets
    for model_name in models:
        model_data = data.get(model_name, {})

        if not model_data:
            print(f"No data found for model: {model_name}. Skipping...")
            continue

        for validation_dataset in validation_datasets:
            val_data = model_data.get(validation_dataset, {})
            if not val_data:
                print(f"No data found for validation dataset: {validation_dataset} in model: {model_name}. Skipping...")
                continue

            # For each species with its own threshold list
            for species, thresholds in species_thresholds.items():
                species_data = val_data.get(species, {})
                
                if not species_data:
                    print(f"No data found for species: {species} in model: {model_name} for validation dataset: {validation_dataset}. Skipping...")
                    continue

                # Reference at threshold 0
                reference_counts_0 = species_data.get('0', {}).get('reference_counts', None)
                if reference_counts_0:
                    dates_0 = sorted(reference_counts_0.keys())
                    dates_0 = [date.split(' ')[0] for date in dates_0]
                    ref_values_0 = [reference_counts_0.get(date, 0) for date in dates_0]
                    all_dates.update(pd.to_datetime(dates_0))

                    fig.add_trace(go.Scatter(
                        x=dates_0,
                        y=ref_values_0,
                        mode='lines',
                        name=f'{species} - Reference (Threshold 0) - {model_name} - {validation_dataset}',
                        line=dict(dash='solid', color=colors[color_index % len(colors)])
                    ))
                    color_index += 1

                # Predictions for requested thresholds
                for threshold in thresholds:
                    if threshold == 0:
                        continue

                    str_threshold = str(threshold)
                    prediction_counts = species_data.get(str_threshold, {}).get('prediction_counts', None)

                    if not prediction_counts:
                        print(f"No prediction data for species: {species} at threshold: {threshold} in model: {model_name}. Skipping...")
                        continue

                    all_dates_threshold = set(reference_counts_0.keys()) | set(prediction_counts.keys()) if reference_counts_0 else set(prediction_counts.keys())
                    dates = sorted([date.split(' ')[0] for date in all_dates_threshold])
                    all_dates.update(pd.to_datetime(dates))

                    pred_values = [prediction_counts.get(date, 0) for date in dates]

                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=pred_values,
                        mode='lines',
                        name=f'{species} - Threshold {threshold} - {model_name} - {validation_dataset}',
                        line=dict(dash='solid', color=colors[color_index % len(colors)])
                    ))
                    color_index += 1

    if not fig.data:
        print("No data to plot. Please check the data structure or the specified models and thresholds.")
        return

    # Global date range for the title
    if all_dates:
        start_date = min(all_dates).strftime('%Y-%m-%d')
        end_date = max(all_dates).strftime('%Y-%m-%d')
    else:
        start_date = "N/A"
        end_date = "N/A"

    species_str = ', '.join(species_thresholds.keys())
    models_str = ', '.join(models)
    datasets_str = ', '.join(validation_datasets)

    title = f"Date Range: {start_date} to {end_date}<br>Models: {models_str}<br>Datasets: {datasets_str}<br>Species: {species_str}"
    print(f"Generated Plot Title:\n{title}")

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', yanchor='top'),
        xaxis_title="Date",
        yaxis_title="Counts",
        template="plotly_white",
        legend=dict(
            title="Legend",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        )
    )

    species_str_filename = '-'.join(species_thresholds.keys()).replace(' ', '_')
    models_str_filename = '-'.join(models).replace(' ', '_')
    datasets_str_filename = '-'.join(validation_datasets).replace(' ', '_')
    save_file_name = f"{datasets_str_filename}_{models_str_filename}_{species_str_filename}_timeseries.html"

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_file_name)
    fig.write_html(save_path)
    print(f"Plot successfully saved at: {save_path}")


def save_all_threshold_retention_tables(data, save_path):
    """
    Build a single CSV ('all_models_threshold_retention.csv') with the percentage of
    event-days retained at each available threshold for every model / dataset / species.

    Baseline threshold = the first (numerically smallest) available threshold for that
    species, which should be '0'. If baseline has 0 event-days, that combo is skipped.
    """
    os.makedirs(save_path, exist_ok=True)

    table_rows = []

    for model_name, model_data in data.items():
        for validation_dataset_name, validation_data in model_data.items():
            for species, species_data in validation_data.items():
                # Numeric sort of string keys; ignore the container key
                available_thresholds = sorted(
                    [threshold for threshold in species_data.keys() if threshold != 'prediction_counts'],
                    key=float
                )
                if not available_thresholds:
                    print(f"No thresholds found for species {species} in validation dataset {validation_dataset_name}")
                    continue

                baseline_threshold = available_thresholds[0]
                total_count = sum(
                    1 for v in species_data.get(baseline_threshold, {}).get('prediction_counts', {}).values()
                    if v is not None and v > 0
                )
                if total_count == 0:
                    print(f"Total count at threshold {baseline_threshold} for species {species} is 0. No data to process.")
                    continue

                for str_threshold in available_thresholds:
                    prediction_counts = species_data.get(str_threshold, {}).get('prediction_counts', {})
                    non_zero_count = sum(1 for v in prediction_counts.values() if v is not None and v > 0)

                    percentage_kept = (non_zero_count / total_count) * 100 if total_count else 0

                    # Force exactly 100% at threshold 0
                    if float(str_threshold) == 0:
                        percentage_kept = 100.0
                    
                    table_rows.append({
                        'Model Name': model_name,
                        'Validation Dataset': validation_dataset_name,
                        'Species': species,
                        'Threshold': float(str_threshold),
                        'Percentage Kept': round(percentage_kept, 2)
                    })

    csv_filename = os.path.join(save_path, "all_models_threshold_retention.csv")
    fieldnames = ['Model Name', 'Validation Dataset', 'Species', 'Threshold', 'Percentage Kept']

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    print(f"Saved threshold retention table for all models at {csv_filename}")


###################################################################################################
# Below: example usage. Update paths to match your environment.
###################################################################################################

# JSON files produced by 01_build_dict_inference.py
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

# Load and combine
dat = load_json_data(json_paths)

# Thresholds to visualize (match those used upstream if possible)
thresholds=[0, 0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]

# Where to save discard plots and CSV summaries
save_dir = '02_data_discard/02.1_data_discarded/'

# 1) Global retention table across all models/species/datasets
save_all_threshold_retention_tables(dat, save_dir)

# 2) Per-dataset "% removed vs threshold" plots
plot_threshold_discard(data=dat, thresholds=thresholds, save_path=save_dir)

# 3) Per-model CSV listing max threshold that keeps ≥ 70%
save_dir = '02_data_discard/02.1_data_discarded/'
save_threshold_discard_table(data=dat, save_path=save_dir,  model_name="2022Q4_14pol_wd_10m", keep_percentage=70)
save_threshold_discard_table(data=dat, save_path=save_dir,  model_name="2025Q1_Beta_onnx", keep_percentage=70)
save_threshold_discard_table(data=dat, save_path=save_dir,  model_name="2024Q4_swisens_First_Stage_v3", keep_percentage=70)
save_threshold_discard_table(data=dat, save_path=save_dir,  model_name="2025Q1_Gamma_onnx", keep_percentage=70)
save_threshold_discard_table(data=dat, save_path=save_dir,  model_name="2025Q2_Omega_onnx", keep_percentage=70)
save_threshold_discard_table(data=dat, save_path=save_dir,  model_name="2025Q2_Omega_onnx_mixed", keep_percentage=70)
save_threshold_discard_table(data=dat, save_path=save_dir,  model_name="2025Q2_Omega_onnx_mixed_15sp", keep_percentage=70)
save_threshold_discard_table(data=dat, save_path=save_dir,  model_name="2025Q1_Beta_onnx_mixed_15sp", keep_percentage=70)
save_threshold_discard_table(data=dat, save_path=save_dir,  model_name="2025Q2_Gamma_onnx_mixed_15sp", keep_percentage=70)
save_threshold_discard_table(data=dat, save_path=save_dir,  model_name="2025Q2_Gamma_onnx_mixed_15sp_redFluo", keep_percentage=70)

# 4) One HTML timeseries per (model, dataset, species)
save_dir = '02_data_discard/02.1_timeseries/'
plot_timeseries(dat, thresholds, save_dir)

# 5) Optional: combinations plot (custom selections)
save_dir = '02_data_discard/02.1_timeseries/combinations/'

'''
# Example: choose species (with their threshold lists), datasets, and models
species_thresholds = {
    "Raindrops": [0, 0.2],
    "Alnus": [0, 0.2],
    "Betula": [0, 0.2],
    "Poaceae": [0, 0.2],
    "Quercus": [0, 0.2],
    "Fagus": [0, 0.2],
    "Corylus": [0, 0.2],
    "Fraxinus": [0, 0.2]
}

plot_timeseries_for_selected_species_and_models(
    data=dat,
    species_thresholds=species_thresholds, 
    validation_datasets=["2024_PBU"], 
    models=['2024Q4_Alpha', '2022Q4_14pol_wd_10m', '2024Q4_swisens_First_Stage_v3', '2025Q1_Beta', '2025Q1_Gamma'], 
    save_dir=save_dir
)
'''

# 6) Optional: build a single combined JSON (handy for later tooling)
combined_data = load_json_data(json_paths)
output_file = "01_json_files/combined_data.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(combined_data, f, indent=4)
