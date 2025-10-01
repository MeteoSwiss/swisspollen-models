# ============================================================
# Compute min/max envelopes across datasets per threshold,
# then integrate (trapezoid) to get area metrics per species/model.
#
# Outputs:
#  - areas_scaling_ratio_kendall.csv : area metrics per species × model
#  - ranges_minmax_diffs.csv         : per-threshold min/max and diffs
# ============================================================

library(dplyr)
library(readr)
library(stringr)

# ---------- 1) Load and clean ----------
# Adjust the path as needed
input_csv <- "SI_2_metrics.csv"
df_raw <- read_csv(input_csv, show_col_types = FALSE)

# Helper: convert possible "comma decimals" to proper numeric
to_num <- function(x) {
  if (is.numeric(x)) return(x)
  x_chr <- as.character(x)
  x_chr <- str_replace_all(x_chr, ",", ".")
  suppressWarnings(as.numeric(x_chr))
}

df <- df_raw %>%
  mutate(
    scaling_factor = to_num(scaling_factor),
    threshold     = to_num(threshold),
    ratio_mean    = to_num(ratio_mean),
    kendall_corr  = to_num(kendall_corr)
  ) %>%
  filter(!is.na(threshold)) %>%
  # drop unwanted models
  filter(!model %in% c("2025Q2_Omega_onnx_mixed", "2024Q4_swisens_First_Stage_v3"))

# ---------- 2) Aggregate min/max across datasets for each threshold ----------
# Note: min/max with na.rm=TRUE returns Inf/-Inf if all values are NA.
# We replace Inf/-Inf with NA afterwards.
safe_min <- function(x) { v <- suppressWarnings(min(x, na.rm = TRUE)); if (is.infinite(v)) NA_real_ else v }
safe_max <- function(x) { v <- suppressWarnings(max(x, na.rm = TRUE)); if (is.infinite(v)) NA_real_ else v }

df_range <- df %>%
  group_by(species, model, threshold) %>%
  summarise(
    min_sf     = safe_min(scaling_factor),
    max_sf     = safe_max(scaling_factor),
    min_ratio  = safe_min(ratio_mean),
    max_ratio  = safe_max(ratio_mean),
    min_kendall= safe_min(kendall_corr),
    max_kendall= safe_max(kendall_corr),
    .groups = "drop"
  ) %>%
  arrange(species, model, threshold)

# Add envelope widths (useful for QC/visualisation)
df_range <- df_range %>%
  mutate(
    diff_sf      = ifelse(is.na(min_sf) | is.na(max_sf), NA_real_, max_sf - min_sf),
    diff_ratio   = ifelse(is.na(min_ratio) | is.na(max_ratio), NA_real_, max_ratio - min_ratio),
    diff_kendall = ifelse(is.na(min_kendall) | is.na(max_kendall), NA_real_, max_kendall - min_kendall)
  )

# ---------- 3) Trapezoid integration over thresholds ----------
# Assumes x is sorted. Removes NA pairs for y before integrating.
trapz_area <- function(x, y) {
  # Keep only finite pairs
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]; y <- y[ok]
  if (length(x) < 2) return(NA_real_)
  # Ensure strictly increasing x; if ties exist, keep first occurrence
  ord <- order(x)
  x <- x[ord]; y <- y[ord]
  # Drop any remaining duplicates in x
  dedup <- !duplicated(x)
  x <- x[dedup]; y <- y[dedup]
  if (length(x) < 2) return(NA_real_)
  sum((x[-1] - x[-length(x)]) * (y[-1] + y[-length(y)]) / 2)
}

df_area <- df_range %>%
  group_by(species, model) %>%
  summarise(
    area_sf_max      = trapz_area(threshold, max_sf),
    area_sf_min      = trapz_area(threshold, min_sf),
    area_sf_diff     = ifelse(is.na(area_sf_max) | is.na(area_sf_min), NA_real_, area_sf_max - area_sf_min),
    
    area_ratio_max   = trapz_area(threshold, max_ratio),
    area_ratio_min   = trapz_area(threshold, min_ratio),
    area_ratio_diff  = ifelse(is.na(area_ratio_max) | is.na(area_ratio_min), NA_real_, area_ratio_max - area_ratio_min),
    
    area_kendall_max = trapz_area(threshold, max_kendall),
    area_kendall_min = trapz_area(threshold, min_kendall),
    area_kendall_diff= ifelse(is.na(area_kendall_max) | is.na(area_kendall_min), NA_real_, area_kendall_max - area_kendall_min),
    .groups = "drop"
  )

# Optionally drop rows where everything is NA
all_area_cols <- c("area_sf_max","area_sf_min","area_sf_diff",
                   "area_ratio_max","area_ratio_min","area_ratio_diff",
                   "area_kendall_max","area_kendall_min","area_kendall_diff")

df_area_clean <- df_area %>%
  filter(rowSums(is.na(across(all_of(all_area_cols)))) < length(all_area_cols))

# ---------- 4) Save outputs ----------
write_csv(df_area_clean, "areas_scaling_ratio_kendall.csv")
write_csv(df_range,      "ranges_minmax_diffs.csv")

# Print a preview
print(df_area_clean)
