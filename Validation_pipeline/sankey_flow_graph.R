# =============================================================================
# Compare two inference models with a Sankey diagram
# =============================================================================
# What this script does
# ---------------------
# 1) Recursively load Parquet outputs for two models (A and B) across locations.
# 2) Keep events that exist in both models (join on uuid).
# 3) Normalize/compact labels (optionally group selected taxa into "Other").
# 4) Build a Sankey diagram (networkD3) showing flows from Model A classes to
#    Model B classes, with node labels including total counts.
# 5) Save the interactive HTML and a table of transition counts.
#
# Notes
# -----
# - Input folders' paths must be adapted to your file system structure.
# - Parquet files are expected to contain columns: uuid, timestamp, <..._preds>, <..._confs>.
# - <..._preds> is detected via suffix "_preds", and <..._confs> via "_confs".
# - Time filtering uses Model A timestamps (you can switch if needed).
# - Unknown/unused classes are grouped into "Other" via the 'to_other' vector.
# - Colors are assigned by taxon/group; any missing mapping gets a neutral fallback.
# =============================================================================

library(arrow)
library(dplyr)
library(tidyr)
library(networkD3)
library(purrr)
library(fs)
library(stringr)
library(lubridate)
library(readr)

# -----------------------------------------------------------------------------
# Input folders (one vector per model)
# -----------------------------------------------------------------------------
dirs_modelA <- c(
  "model2025/data/poleno-30/2024_PPY/joined/2022Q4_14pol_wd_10m/year=2024/",
  "model2025/data/poleno-13/2024_PBU/joined/2022Q4_14pol_wd_10m/year=2024/",
  "model2025/data/poleno-18/2024_PNE/joined/2022Q4_14pol_wd_10m/year=2024/",
  "model2025/data/poleno-21/2024_PLZ/joined/2022Q4_14pol_wd_10m/year=2024/",
  "model2025/data/poleno-27/2024_PBS/joined/2022Q4_14pol_wd_10m/year=2024/"
)
dirs_modelB <- c(
  "model2025/data/poleno-30/2024_PPY/joined/2025Q2_Gamma_onnx_mixed_15sp/year=2024/",
  "model2025/data/poleno-13/2024_PBU/joined/2025Q2_Gamma_onnx_mixed_15sp/year=2024/",
  "model2025/data/poleno-18/2024_PNE/joined/2025Q2_Gamma_onnx_mixed_15sp/year=2024/",
  "model2025/data/poleno-21/2024_PLZ/joined/2025Q2_Gamma_onnx_mixed_15sp/year=2024/",
  "model2025/data/poleno-27/2024_PBS/joined/2025Q2_Gamma_onnx_mixed_15sp/year=2024/"
)

# -----------------------------------------------------------------------------
# Helper: read all Parquet files under a root folder for one model
# -----------------------------------------------------------------------------
read_model_parquets <- function(root_dir, model_label) {
  parquet_files <- fs::dir_ls(root_dir, recurse = TRUE, regexp = "\\.parquet$")
  message(length(parquet_files), " parquet files found for model: ", model_label)
  
  data_list <- map(parquet_files, function(f) {
    df <- tryCatch(read_parquet(f), error = function(e) NULL)
    if (is.null(df)) return(NULL)
    
    cols <- colnames(df)
    uuid_col <- "uuid"
    pred_col <- grep("_preds$", cols, value = TRUE)
    conf_col <- grep("_confs$", cols, value = TRUE)
    
    # Expect exactly one preds/conf column and a uuid column
    if (length(pred_col) != 1 || length(conf_col) != 1 || !(uuid_col %in% cols)) {
      warning("Skipping file (missing or ambiguous columns): ", f)
      return(NULL)
    }
    
    df %>%
      select(uuid, timestamp, class = all_of(pred_col), conf = all_of(conf_col)) %>%
      rename(
        !!paste0("class_", model_label) := class,
        !!paste0("conf_", model_label) := conf,
        !!paste0("timestamp_", model_label) := timestamp
      )
  })
  
  bind_rows(Filter(Negate(is.null), data_list))
}

# -----------------------------------------------------------------------------
# Read all folders for each model
# -----------------------------------------------------------------------------
read_multiple_dirs <- function(dirs, model_label) {
  map_dfr(dirs, ~ read_model_parquets(.x, model_label))
}

df_A <- read_multiple_dirs(dirs_modelA, "A")
df_B <- read_multiple_dirs(dirs_modelB, "B")

# -----------------------------------------------------------------------------
# Inner-join on uuid to keep events scored by both models
# -----------------------------------------------------------------------------
df_compare <- inner_join(df_A, df_B, by = "uuid")
message("Events with both model predictions: ", nrow(df_compare))

# -----------------------------------------------------------------------------
# Time filtering (based on Model A timestamp)
# -----------------------------------------------------------------------------
# Example alternate periods:
# filter_start <- as.POSIXct("2024-01-01 00:00:00", tz = "UTC")
# filter_end   <- as.POSIXct("2024-03-15 23:59:59", tz = "UTC")
# or:
# filter_start <- as.POSIXct("2024-05-01 00:00:00", tz = "UTC")
# filter_end   <- as.POSIXct("2024-09-30 23:59:59", tz = "UTC")

# Whole year window used here
filter_start <- as.POSIXct("2024-01-01 00:00:00", tz = "UTC")
filter_end   <- as.POSIXct("2024-09-30 23:59:59", tz = "UTC")

df_compare <- df_compare %>%
  mutate(timestamp = as_datetime(timestamp_A, tz = "UTC")) %>%
  filter(timestamp >= filter_start, timestamp <= filter_end)

message("Filtered events in range: ", nrow(df_compare))

# -----------------------------------------------------------------------------
# Label harmonization and grouping into "Other"
# -----------------------------------------------------------------------------
# Any class in 'to_other' will be relabeled as "Other"
to_other <- c(
  "Fagus", "Taxus", "Ulmus", "Poaceae", "Quercus", "Populus",
  "Carpinus", "Fraxinus", "Pinaceae", "Platanus", "Cupressus", "Water"
)

df_compare <- df_compare %>%
  mutate(
    class_A = str_to_title(str_to_lower(class_A)),
    class_B = str_to_title(str_to_lower(class_B)),
    class_A = ifelse(class_A %in% to_other, "Other", class_A),
    class_B = ifelse(class_B %in% to_other, "Other", class_B)
  )

# -----------------------------------------------------------------------------
# Build flows table (counts from A -> B)
# -----------------------------------------------------------------------------
flows <- df_compare %>%
  count(class_A, class_B, name = "value") %>%
  mutate(
    source = paste0("A: ", class_A),
    target = paste0("B: ", class_B)
  ) %>%
  select(source, target, value)

# -----------------------------------------------------------------------------
# Build nodes with total counts and group labels
# -----------------------------------------------------------------------------
node_names <- unique(c(flows$source, flows$target))
nodes <- data.frame(
  name  = node_names,
  label = node_names,    # will be updated with counts
  stringsAsFactors = FALSE
)

# Total incoming/outgoing counts per node for labeling
source_counts <- flows %>% group_by(source) %>% summarise(count = sum(value), .groups = "drop")
target_counts <- flows %>% group_by(target) %>% summarise(count = sum(value), .groups = "drop")
all_counts <- bind_rows(
  source_counts %>% rename(name = source),
  target_counts %>% rename(name = target)
) %>%
  group_by(name) %>%
  summarise(count = sum(count), .groups = "drop")

# Add counts; extract taxon name as group (drop "A: " / "B: " prefix)
nodes <- nodes %>%
  left_join(all_counts, by = "name") %>%
  mutate(
    count = ifelse(is.na(count), 0L, count),
    label = paste0(name, " (", count, ")"),
    class = str_remove(name, "^A: |^B: "),
    group = class
  )

# Map flows to node indices
flows$source_id <- match(flows$source, nodes$name) - 1
flows$target_id <- match(flows$target, nodes$name) - 1

# -----------------------------------------------------------------------------
# Colors per group (fallback for any group not listed)
# -----------------------------------------------------------------------------
color_map <- c(
  "Water"     = "#1f77b4",
  "Poaceae"   = "#00AFB9",
  "Fagus"     = "#32DE8A",
  "Fraxinus"  = "#9467bd",
  "Alnus"     = "#7EBC89",
  "Taxus"     = "#779CAB",
  "Populus"   = "#FFD4CA",
  "Corylus"   = "#896978",
  "Ulmus"     = "#FED9B7",
  "Cupressus" = "#D3FFE9",
  "Other"     = "#7f7f7f",
  "Pinaceae"  = "#F9C80E",
  "Quercus"   = "#EA3546",
  "Betula"    = "#2ca02c",
  "Carpinus"  = "#8c564b",
  "Platanus"  = "#e377c2"
)

groups <- unique(nodes$group)
colors <- unname(color_map[groups])
# Fallback color for groups not in color_map
colors[is.na(colors)] <- "#999999"

color_scale <- paste0(
  "d3.scaleOrdinal().domain([\"",
  paste(groups, collapse = "\", \""), "\"])",
  ".range([\"",
  paste(colors, collapse = "\", \""), "\"])"
)

# -----------------------------------------------------------------------------
# Sankey diagram
# -----------------------------------------------------------------------------
sankey <- sankeyNetwork(
  Links = flows,
  Nodes = nodes,
  Source = "source_id",
  Target = "target_id",
  Value = "value",
  NodeID = "label",      # display label (with counts)
  NodeGroup = "group",   # color by group/taxon
  fontSize = 12,
  nodeWidth = 30,
  sinksRight = FALSE,
  colourScale = color_scale
)

sankey
# -----------------------------------------------------------------------------
# Export: HTML widget and transition table
# -----------------------------------------------------------------------------
library(htmlwidgets)
# If you need PNG export via webshot, install PhantomJS once:
# webshot::install_phantomjs()
# library(webshot)

html_path <- "/path/to/output.html"
saveWidget(sankey, html_path, selfcontained = TRUE)

transitions_all <- df_compare %>%
  count(class_A, class_B, sort = TRUE, name = "n")

write_csv(
  x = transitions_all,
  file = "/path/to/output.csv"
)

message("Saved Sankey HTML to: ", html_path)
message("Saved transition counts.)
