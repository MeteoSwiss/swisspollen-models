"""
02.2_plot_boxplots_thresholds.py
==========================================

Purpose
-------
Create an interactive boxplot (HTML) that compares the confidence thresholds
across multiple models, for each species (and implicitly across validation datasets).

Where outputs go
----------------
02_data_discard/02.2_boxplot_discarded

Inputs
------
This script expects the **per-model CSVs** produced by:
    save_threshold_discard_table(data, save_path, model_name=..., keep_percentage=70)

Each CSV typically contains columns:
    - Model Name
    - Validation Dataset
    - Species
    - Threshold


What the figure shows
---------------------
- One box per (Species – Model).
- The distribution comes from all rows found for that (Species, Model) across the model’s CSV,
  which typically represents multiple validation datasets.

Quick start
-----------
1) Make sure you have the per-model CSV files listed in `model_csv_list`.
2) Install dependencies (preferably in a virtual environment):
       pip install pandas plotly
3) Run:
       python 02.2_plot_boxplots_thresholds.py
4) Open the generated HTML file in a browser.

Notes
-----
- Axis titles and the main title are in French (kept as-is to avoid code changes).
- This script does not define or control the period covered by the JSON timelines.
  Any time-range trimming is handled upstream when building or trimming the JSONs.

"""

import pandas as pd
import plotly.graph_objects as go
import os


def create_comparison_boxplot(model_csv_list, output_html):
    """
    Build a single interactive HTML boxplot to compare thresholds between models,
    for each species.

    Parameters
    ----------
    model_csv_list : list of str
        Paths to the per-model CSV files (produced by save_threshold_discard_table).
        Each row contributes a threshold value to the corresponding (Species, Model) box.
    output_html : str
        Path where the HTML figure will be written.
    """
    # Collect rows from all provided CSV files
    all_data = []

    # Read each CSV and tag its rows with a Model identifier taken from the filename
    for model_csv in model_csv_list:
        df = pd.read_csv(model_csv)
        model_name = os.path.basename(model_csv).split('.')[0]  # Derive model label from filename
        df['Model'] = model_name  # Add a helper column to identify which model this row belongs to
        all_data.append(df)

    # Combine all rows into one DataFrame
    df_combined = pd.concat(all_data, ignore_index=True)

    # All species found across all CSVs
    species_list = df_combined['Species'].unique()

    # Create an empty Plotly figure
    fig = go.Figure()

    # For each species, plot one box per model showing the distribution of thresholds
    for species in species_list:
        species_data = df_combined[df_combined['Species'] == species]

        for model in species_data['Model'].unique():
            model_species_data = species_data[species_data['Model'] == model]['Threshold']

            fig.add_trace(go.Box(
                y=model_species_data,
                name=f"{species} - {model}",
                boxmean='sd'  # Show mean and standard deviation line on the box
            ))

    # Figure layout (titles and labels kept in French)
    fig.update_layout(
        title="Comparaison des seuils entre modèles pour chaque espèce",
        xaxis_title="Espèce - Modèle",
        yaxis_title="Seuil",
        showlegend=True
    )

    # Write the interactive figure to an HTML file (open it in any browser)
    fig.write_html(output_html)
    print(f"Boxplot comparatif enregistré sous {output_html}")


# Example usage (edit the file paths to match your environment):
model_csv_list = [
    '02_data_discard/02.1_data_discarded/2025Q1_Beta_onnx_mixed_15sp_thresholds_to_keep_70pct.csv',
    '02_data_discard/02.1_data_discarded/2025Q2_Gamma_onnx_mixed_15sp_redFluo_thresholds_to_keep_70pct.csv'
]

output_html = '02_data_discard/02.2_boxplot_discarded/threshold_comparison_boxplot.html'

# Run the boxplot creation
create_comparison_boxplot(model_csv_list, output_html)
