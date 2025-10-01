library(readxl)
library(dplyr)
library(tibble)

# Load an Excel file with a customizable header row
load_excel_file <- function(path, header_row = 1) {
  raw <- read_excel(path, col_names = FALSE)
  col_names <- as.character(unlist(raw[header_row, ]))
  df <- raw[-c(1:header_row), ]
  colnames(df) <- make.unique(col_names)
  df <- as_tibble(df)
  
  # Keep the first column as character
  df[[1]] <- trimws(as.character(df[[1]]))
  colnames(df)[1] <- "date"
  
  return(df)
}

# Load Excel files
df1 <- load_excel_file("/path/to/P2.xlsx", header_row = 3)
df2 <- load_excel_file("/path/to/P4.xlsx", header_row = 1)
df3 <- load_excel_file("/path/to/P8.xlsx", header_row = 1)

# Extract common dates (character format)
common_dates <- Reduce(intersect, list(df1$date, df2$date, df3$date))

# Filter the three data frames to keep only common dates
df1_f <- df1 %>% filter(date %in% common_dates)
df2_f <- df2 %>% filter(date %in% common_dates)
df3_f <- df3 %>% filter(date %in% common_dates)

# Find common column names (excluding "date")
common_cols <- Reduce(intersect, list(names(df1_f), names(df2_f), names(df3_f)))
common_cols <- setdiff(common_cols, "date")

# Function to extract only the taxon name from a column name
extract_taxon <- function(colname) {
  gsub(".*de\\s+", "", colname)
}

# Clean numeric columns: convert values to numeric, replace commas with dots,
# and remove non-numeric characters
clean_numeric_cols <- function(df) {
  df %>%
    mutate(across(-date, ~ suppressWarnings(as.numeric(gsub(",", ".", gsub("[^0-9\\.\\-]", "", .))))))
}

df1_f <- clean_numeric_cols(df1_f)
df2_f <- clean_numeric_cols(df2_f)
df3_f <- clean_numeric_cols(df3_f)

# Compute Kendall Tau correlations between pairs of data frames
calculate_tau <- function(col) {
  x1 <- df1_f[[col]]
  x2 <- df2_f[[col]]
  x3 <- df3_f[[col]]
  
  tau12 <- if (all(is.na(x1)) || all(is.na(x2))) NA else cor(x1, x2, method = "kendall", use = "pairwise.complete.obs")
  tau13 <- if (all(is.na(x1)) || all(is.na(x3))) NA else cor(x1, x3, method = "kendall", use = "pairwise.complete.obs")
  tau23 <- if (all(is.na(x2)) || all(is.na(x3))) NA else cor(x2, x3, method = "kendall", use = "pairwise.complete.obs")
  
  data.frame(
    taxon = extract_taxon(col),
    df_pair = c("df1_vs_df2", "df1_vs_df3", "df2_vs_df3"),
    kendall_tau = c(tau12, tau13, tau23)
  )
}

# Build the final results data frame
results <- do.call(rbind, lapply(common_cols, calculate_tau))
print(results)

# Keep only a subset of taxa
taxons_to_keep <- c("Alnus", "Betula", "Corylus", "Fagus", "Fraxinus", "Poaceae", "Quercus")

# Filter the results for the selected taxa
results_filtered <- results %>% 
  filter(taxon %in% taxons_to_keep)

# Compute the mean Kendall Tau for each taxon and sort by descending value
tau_by_taxon <- results_filtered %>%
  group_by(taxon) %>%
  summarise(mean_kendall_tau = mean(kendall_tau, na.rm = TRUE)) %>%
  arrange(desc(mean_kendall_tau))
