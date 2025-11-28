## Introduction ----------------------------------------------------------------
# Author: Isabel Kitschelt (m1ixk00)
# Purpose: Create tidy data
rm(list = ls()); gc()
require(pacman)
pacman::p_load(
  tm, tidytext, # text
  tidyverse, data.table, # Data cleaning
  zoo, lubridate, # Dates
  ggplot2, cowplot, ggcorrplot, stargazer, # Plotting
  readxl, # Read excel
  fredr, httr, rvest # Webscraping & Fred
)

# SETUP ------------------------------------------------------------------------
## Load tidy Beige Book data
final_data <- fread("./data/final_data_with_headers.csv", header=TRUE) %>%
  select(-c(V1, Index))
unique(final_data$bank) # make sure National Summary is not included

# TIDY DATA --------------------------------------------------------------------
final_data_tidy = custom_tokenizer(final_data)

final_data_tidy <- final_data_tidy %>%
  dplyr::select(pub_date, bank, word)

# SAVE -------------------------------------------------------------------------
write.csv(final_data_tidy, "/beige_book_nlp/data/tidy_data_with_headers.csv", row.names = FALSE)
write.csv(final_data_tidy, "/collaboration/beige_books/data/tidy_data_with_headers.csv", row.names = FALSE)
