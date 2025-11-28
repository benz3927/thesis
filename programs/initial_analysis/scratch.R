# Introduction ---------------------------------------------------------------------------------------------------
# Script purpose/description: Create Fed Funds Rate and Gender Percentage Chart for Transcript Analysis Paper
# *_ Once we get BoardEx Data
rm(list=ls()); cat("\014")


#Bring in data
#The data I will give you will be a list of the transcripts that are first name and last name of the speaker
#You will use the BoardEx data to identify the gender of the speaker. The gender categories we will be using are the ones previously used in the transcripts before 2009 - MR | MS | SPEAKER | GOVERNOR
#Because the BoardEx dat aiwll be available on Tuesday, the 20th, we will work on settingup the FRED API to gather the interest rates Professor Owen wanted to see


#Pull from FRED
library(fredr)
library(pkgconfig)
library(httr)


#You need to first create an account here and collect an API key

key <- "0d56cc2ebb26c63195548e1a4cace451"

FRED_API_KEY <- fredr_set_key(key)
FFER_FRED <- fredr_series_observations(series_id = "DFF",
                                       observation_start = as.Date('1994-01-01'),
                                       observation_end = Sys.Date())
FFER_FRED <- select(FFER_FRED, subset = -c(realtime_start, realtime_end))
#---Graph 1 Make a line chart of the federal funds rate and Percentage of Males to Female
total <- read.csv('./output/total_clean.csv') #Percent can be charted until and through December 2008
#ifgraphics: https://figs.web.rsma.frb.gov/

total$date <- as.Date(gsub("(\\d{4})(\\d{2})(\\d{2})$","\\1-\\2-\\3",total$date))
merged <- merge(FFER_FRED, total, by = ("date"), all = TRUE)
merged.df <- merged %>% filter(date <= "2009-01-01") %>% filter(!is.na(Percent.Male.Att))


scale = 0.25
ggplot(data = merged %>% filter(date <= "2009-01-01") %>% filter(!is.na(Percent.Male.Att)), aes(x=date, y = value)) + 
  geom_line(color = "black") + 
  geom_line(aes(y = Percent.Male.Att/scale), color="blue") +
  scale_y_continuous(sec.axis = sec_axis(~.*scale, name="blue")) +
  labs(x = "Date", y = "DFF/Percent Male")

max_first  <- max(merged.df$value)   # Specify max of first y axis
max_second <- max(merged.df$Percent.Male.Att) # Specify max of second y axis
min_first  <- min(merged.df$value)   # Specify min of first y axis
min_second <- min(merged.df$Percent.Male.Att) # Specify min of second y axis

# scale and shift variables calculated based on desired mins and maxes
scale = (max_second - min_second)/(max_first - min_first)
scale = 0.025
shift = min_first - min_second

# Function to scale secondary axis
scale_function <- function(x, scale, shift){
  return ((x)*scale - shift)
}
# Function to scale secondary variable values
inv_scale_function <- function(x, scale, shift){
  return ((x + shift)/scale)
}

chair_starts <- c(as.numeric(as.Date("1987-08-11")), as.numeric(as.Date("2006-02-01")), as.numeric(as.Date("2014-02-03")), as.numeric(as.Date("2018-02-05")))

ggplot(data = merged.df, aes(x=date, y = value)) + 
  geom_line(aes(color = "DFF")) + 
  geom_line(aes(y = inv_scale_function(Percent.Male.Att, scale, shift-0.16), color="Percent Males")) +
  scale_y_continuous(limits = c(min_first, max_first), 
        sec.axis = sec_axis(~scale_function(., scale, shift-0.16), name="Percent Males")) +
  labs(x = "Date", y = "DFF", color = "") +
  geom_vline(xintercept = chair_starts, linetype="dashed")


# ggplot(data = merged.df, aes(x=date, y = value)) + 
#   geom_line(aes(color = "DFF")) + 
#   geom_line(aes(y = inv_scale_function(Percent.Male.Att, scale, shift), color="Percent Males")) +
#   scale_y_continuous(
#     limits = c(0, 8), #left-hand axis 
#     breaks = seq(0, 8, 2), #left-hand axis
#     sec.axis = dup_axis(~.* (1 / scale), #right-hand-axis-your scaling factor is what you created in the function
#                         # name = "Share of world GDP",
#                         breaks = seq(0.7, 1, 0.05)), #breaks for right-hand-axis
#     expand = c(0, 0)) +
#   #scale_y_continuous(limits = c(min_first, max_first), 
#    #                  sec.axis = sec_axis(~scale_function(., scale, shift), name="Percent Males")) +
#   labs(x = "Date", y = "DFF", color = "")


#----Graph #2 Laughs vs. DFF 
merged.laughs <- merged %>% filter(!is.na(All.laughs))

max_first  <- max(merged.laughs$value)   # Specify max of first y axis
max_second <- max(merged.laughs$All.laughs) # Specify max of second y axis
min_first  <- min(merged.laughs$value)   # Specify min of first y axis
min_second <- min(merged.laughs$All.laughs) # Specify min of second y axis

# scale and shift variables calculated based on desired mins and maxes
scale = (max_second - min_second)/(max_first - min_first)
shift = min_first - min_second

# Function to scale secondary axis
scale_function <- function(x, scale, shift){
  return ((x)*scale - shift)
}
# Function to scale secondary variable values
inv_scale_function <- function(x, scale, shift){
  return ((x + shift)/scale)
}
ggplot(data = merged.laughs, aes(x=date, y = value)) + 
  geom_line(aes(color = "Federal Funds Rate")) + 
  geom_line(aes(y = inv_scale_function(All.laughs, scale, shift), color="Laughs")) +
  scale_y_continuous(limits = c(min_first, max_first), 
                     sec.axis = sec_axis(~scale_function(., scale, shift), name="Laughs")) +
  labs(x = "Date", y = "Federal Funds Rate", color = "") +
  geom_vline(xintercept = chair_starts, linetype="dashed")



#----Graph 3-4 Finish off attendance graphs*





