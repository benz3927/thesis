## Intro -----------------------------------------------------------------------
# Author: Dylan Saez
# Date: April 26, 2023
# Script purpose/description: Webscrape FOMC transcripts

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
source("/fomc_transcript/programs/helpers/custom_tokenizer.R")

setwd("/fomc_transcript/")

## Read manual input -----------------------------------------------------------
#Here, put all the new dates of material you want - On January 24, 2024 - I added the 2018 public fates and removed all the rest
pub_dates <- read_excel("./data/metadata/manual_metadata.xlsx") 

## Clean -----------------------------------------------------------------------
# Separate year, month, day info
## Create URLs -----------------------------------------------------------------
pub_dates$transcript_date <- paste(substr(pub_dates$Date,1,4), substr(pub_dates$Date,5,6), substr(pub_dates$Date,7,8), sep = "-")

ending_url_word <- "meeting.pdf"
pub_dates$URL_ends <- paste(pub_dates$Date, ending_url_word, sep = "")

pub_dates <- pub_dates %>%
  separate(transcript_date, into = c("year", "month", "day"), sep = "-") %>%
  mutate(
    year = as.numeric(year), 
    month = as.numeric(month), 
    day = as.numeric(day)
  ) 
pub_dates$transcript_date <- paste(substr(pub_dates$Date,1,4), substr(pub_dates$Date,5,6), substr(pub_dates$Date,7,8), sep = "-")
#
#---
# Create month map
month_map = data.table::data.table(
  month = c(1,2,3,4,5,6,7,8,9,10,11,12),
  month_name = c("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
)

# Merge data to month map
pub_dates <- merge(
  pub_dates, month_map,
  by = "month", 
  all.x = T
) 

# Additional cleaning
pub_dates <- pub_dates %>%
  mutate( # For months 1-9, add "0" before and make month character
    month = ifelse(month %in% c(1,2,3,4,5,6,7,8,9), paste0("0", month), as.character(month)))
  # ) %>%
  # select(-day) # Remove day variable (not needed)

## Create URLs -----------------------------------------------------------------
base_url = "https://www.federalreserve.gov/monetarypolicy/files/FOMC"

full_url_data <- data.table()
for (i in 1:nrow(pub_dates)) {
  
  year_i = pub_dates[["year"]][i]
  month_i = pub_dates[["month"]][i]
  
  date_url = paste0(base_url,pub_dates$URL_ends, "")

  # Create URLs with dates
  
    
    # Add to full data
    full_url_data$url <- rbind(date_url)
    full_url_data$date <- rbind(pub_dates$transcript_date)
    full_url_data$year <- rbind(pub_dates$year)
    full_url_data$month <- rbind(pub_dates$month)
    full_url_data$day <- rbind(pub_dates$day)
    full_url_data$transcript_date <- rbind(pub_dates$transcript_date)
}

## Special cases;
# April 2016 National Summary
# full_url_data <- full_url_data[, url := ifelse(year == 2016 & month == "04" & bank == "National Summary", "https://www.minneapolisfed.org/beige-book-reports/2016/2016-04-national-summary", url)]
# # June 2016 National Summary
# full_url_data <- full_url_data[, url := ifelse(year == 2016 & month == "06" & bank == "National Summary", "https://www.minneapolisfed.org/beige-book-reports/2016/2016-06-national-summary", url)]


write.csv(data.table(full_url_data), "./data/metadata/full_url_data_2.csv")
