# Introduction ---------------------------------------------------------------------------------------------------
# Script purpose/description: Create Fed Funds Rate and Gender Percentage Chart for Transcript Analysis Paper
# *_ Once we get BoardEx Data
rm(list=ls()); cat("\014")

setwd("fomc_transcript/")




#Bring in data
#The data I will give you will be a list of the transcripts that are first name and last name of the speaker
#You will use the BoardEx data to identify the gender of the speaker. The gender categories we will be using are the ones previously used in the transcripts before 2009 - MR | MS | SPEAKER | GOVERNOR
#Because the BoardEx dat aiwll be available on Tuesday, the 20th, we will work on settingup the FRED API to gather the interest rates Professor Owen wanted to see


#Pull from FRED
library(fredr)
library(pkgconfig)
library(httr)

set_config(use_proxy('wwwproxy.frb.gov',8080), override = FALSE)
set_config(user_agent('Lynx'), override = FALSE)

#You need to first create an account here and collect an API key

key <- "PLACE API KEY HERE"

FRED_API_KEY <- fredr_set_key(key)
FFER_FRED <- fredr_series_observations(series_id = "DFF",
                                      observation_start = as.Date('1994-01-01'),
                                      observation_end = Sys.Date())
FFER_FRED <- select(FFER_FRED, subset = -c(realtime_start, realtime_end))
#---Graph 1 Make a line chart of the federal funds rate and Percentage of Males to Female
total <- read.csv('./output/total_clean.csv') #Percent can be charted until and through December 2008
#ifgraphics: https://figs.web.rsma.frb.gov/



#----Graph #2 Laughs vs. DFF 


  
  
#----Graph 3-4 Finish off attendance graphs*





